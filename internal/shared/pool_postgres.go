package shared

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	concurrentmap "github.com/Dsouza10082/ConcurrentOrderedMap"
	"github.com/Dsouza10082/orus/config"
	"github.com/jmoiron/sqlx"
	_ "github.com/jackc/pgx/v5/stdlib"
)

func GetPostgresConnectionInstance() *PostgresConnectionInstance {
	postgresOnce.Do(func() {
		postgresInstance = newPostgresConnectionInstance()
	})
	return postgresInstance
}

func newPostgresConnectionInstance() *PostgresConnectionInstance {
	params := config.GetParameters()
	connInst := &PostgresConnectionInstance{
		created:              time.Now(),
		connections:          concurrentmap.NewConcurrentOrderedMapWithCapacity[string, *PooledPostgresConnection](params.PostgresMaxConnections),
		availableCount:       0,
		PgDBHost:             params.PostgresHost,
		PgMinConnections:     params.PostgresMinConnections,
		PgMaxConnections:     params.PostgresMaxConnections,
		PgConnectionLifeTime: params.PostgresConnectionLifeTime,
		expireConnectionsCh:  make(chan bool, 1),
		renewConnectionsCh:   make(chan bool, 1),
		stopCh:               make(chan bool),
		connectionCounter:    0,
		Verbose:              false,
	}

	for i := 0; i < params.PostgresMinConnections; i++ {
		conn, err := connInst.createConnection()
		if err != nil {
			log.Printf("Error creating initial PostgreSQL connection %d: %v", i, err)
			continue
		}
		connInst.connections.Set(conn.ID, conn)
		connInst.availableCount++
	}

	connInst.wg.Add(1)
	go connInst.maintainPool()

	log.Printf("PostgreSQL connection pool initialized with %d connections", connInst.connections.Len())
	return connInst
}

func (c *PostgresConnectionInstance) createConnection() (*PooledPostgresConnection, error) {
	db, err := sqlx.Open("pgx", c.PgDBHost)
	if err != nil {
		return nil, fmt.Errorf("error opening connection: %w", err)
	}

	db.SetMaxOpenConns(c.PgMaxConnections)
	db.SetMaxIdleConns(c.PgMinConnections)
	db.SetConnMaxLifetime(time.Duration(c.PgConnectionLifeTime) * time.Second)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("error pinging connection: %w", err)
	}

	c.mu.Lock()
	c.connectionCounter++
	id := fmt.Sprintf("pg_conn_%d_%d", c.connectionCounter, time.Now().UnixNano())
	c.mu.Unlock()

	return &PooledPostgresConnection{
		DB:        db,
		CreatedAt: time.Now(),
		LastUsed:  time.Now(),
		ID:        id,
		InUse:     false,
	}, nil
}

func (c *PostgresConnectionInstance) maintainPool() {
	defer c.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			log.Println("Stopping pool maintenance for PostgreSQL")
			return

		case <-ticker.C:
			c.performMaintenance()

		case <-c.expireConnectionsCh:
			log.Println("Request to expire old connections for PostgreSQL")
			c.expireOldConnections()
			c.ensureMinimumConnections()

		case <-c.renewConnectionsCh:
			log.Println("Request to renew connections for PostgreSQL")
			c.ensureMaximumConnections()
		}
	}
}

func (c *PostgresConnectionInstance) performMaintenance() {
	expiredCount := 0
	validCount := 0
	inUseCount := 0
	availableCount := 0

	orderedPairs := c.connections.GetOrderedV2()
	toRemove := make([]string, 0)

	for _, pair := range orderedPairs {
		conn := pair.Value

		if !c.IsConnectionValid(conn) {
			conn.DB.Close()
			toRemove = append(toRemove, pair.Key)
			expiredCount++
		} else {
			validCount++
			if conn.InUse {
				inUseCount++
			} else {
				availableCount++
			}
		}
	}

	for _, id := range toRemove {
		c.connections.Delete(id)
	}

	c.availableCount = int32(availableCount)

	if c.Verbose {
		log.Printf("PostgreSQL Pool Status - Total: %d, Valid: %d, Available: %d, In use: %d, Expired: %d",
			c.connections.Len(), validCount, availableCount, inUseCount, expiredCount)
	}

	if validCount < c.PgMinConnections {
		go c.ensureMinimumConnections()
	}
}

func (c *PostgresConnectionInstance) expireOldConnections() {
	connectionLifeTime := time.Duration(c.PgConnectionLifeTime) * time.Second
	toRemove := make([]string, 0)

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value
		if time.Since(conn.CreatedAt) > connectionLifeTime {
			conn.DB.Close()
			toRemove = append(toRemove, pair.Key)
			if c.Verbose {
				log.Printf("PostgreSQL Connection %s forcibly expired", conn.ID)
			}
		}
	}

	for _, id := range toRemove {
		c.connections.Delete(id)
	}
}

func (c *PostgresConnectionInstance) ensureMinimumConnections() {
	currentCount := c.connections.Len()
	needed := c.PgMinConnections - currentCount

	if needed <= 0 {
		return
	}

	if c.Verbose {
		log.Printf("Creating %d PostgreSQL connections to reach minimum", needed)
	}

	for i := 0; i < needed; i++ {
		conn, err := c.createConnection()
		if err != nil {
			if c.Verbose {
				log.Printf("Error creating replacement PostgreSQL connection: %v", err)
			}
			continue
		}

		c.connections.Set(conn.ID, conn)
		c.availableCount++
		if c.Verbose {
			log.Printf("New PostgreSQL connection %s added to pool", conn.ID)
		}
	}
}

func (c *PostgresConnectionInstance) ensureMaximumConnections() {
	c.mu.Lock()
	defer c.mu.Unlock()

	currentCount := c.connections.Len()
	needed := c.PgMaxConnections - currentCount

	if needed <= 0 {
		if c.Verbose {
			log.Printf("Pool already at maximum (%d PostgreSQL connections)", currentCount)
		}
		return
	}

	if c.Verbose {
		log.Printf("Creating %d additional PostgreSQL connections", needed)
	}

	for i := 0; i < needed; i++ {
		currentCount = c.connections.Len()
		if currentCount >= c.PgMaxConnections {
			if c.Verbose {
				log.Printf("Reached maximum PostgreSQL connections during creation")
			}
			break
		}

		c.mu.Unlock()
		conn, err := c.createConnection()
		c.mu.Lock()

		if err != nil {
			log.Printf("Error creating additional PostgreSQL connection: %v", err)
			continue
		}

		currentCount = c.connections.Len()
		if currentCount >= c.PgMaxConnections {
			conn.DB.Close()
			if c.Verbose {
				log.Printf("Maximum PostgreSQL connections reached, closing excess connection %s", conn.ID)
			}
			break
		}

		c.connections.Set(conn.ID, conn)
		c.availableCount++
		if c.Verbose {
			log.Printf("Additional PostgreSQL connection %s created", conn.ID)
		}
	}
}

func (c *PostgresConnectionInstance) IsConnectionValid(conn *PooledPostgresConnection) bool {
	if conn == nil || conn.DB == nil {
		return false
	}

	connectionLifeTime := time.Duration(c.PgConnectionLifeTime) * time.Second
	if time.Since(conn.CreatedAt) > connectionLifeTime {
		if c.Verbose {
			log.Printf("PostgreSQL Connection %s expired due to lifetime", conn.ID)
		}
		return false
	}

	maxIdleTime := 5 * time.Minute
	if !conn.InUse && time.Since(conn.LastUsed) > maxIdleTime {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		if err := conn.DB.PingContext(ctx); err != nil {
			if c.Verbose {
				log.Printf("PostgreSQL Connection %s failed ping: %v", conn.ID, err)
			}
			return false
		}
	}

	return true
}

func (c *PostgresConnectionInstance) GetConnection(ctx context.Context) (*sqlx.DB, error) {
	c.mu.RLock()
	maxConnections := c.PgMaxConnections
	c.mu.RUnlock()

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value

		if !conn.InUse {
			err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledPostgresConnection) error {
				if (*v).InUse {
					return errors.New("PostgreSQL connection already in use")
				}

				if !c.IsConnectionValid(*v) {
					return errors.New("invalid PostgreSQL connection")
				}

				(*v).InUse = true
				(*v).LastUsed = time.Now()
				c.availableCount--
				return nil
			})

			if err == nil {
				if c.Verbose {
					log.Printf("PostgreSQL Connection %s obtained from pool", conn.ID)
				}
				return conn.DB, nil
			}
		}
	}

	currentCount := c.connections.Len()
	if currentCount < maxConnections {
		c.mu.Lock()
		currentCount = c.connections.Len()
		if currentCount < c.PgMaxConnections {
			c.mu.Unlock()
			newConn, err := c.createConnection()
			c.mu.Lock()

			if err != nil {
				c.mu.Unlock()
				return nil, fmt.Errorf("pool exhausted and could not create new PostgreSQL connection: %w", err)
			}

			currentCount = c.connections.Len()
			if currentCount >= c.PgMaxConnections {
				c.mu.Unlock()
				newConn.DB.Close()
				if c.Verbose {
					log.Printf("Maximum PostgreSQL connections reached while creating new connection, closing %s", newConn.ID)
				}
			} else {
				newConn.InUse = true
				newConn.LastUsed = time.Now()
				c.connections.Set(newConn.ID, newConn)
				actualCount := c.connections.Len()
				c.mu.Unlock()

				if c.Verbose {
					log.Printf("New PostgreSQL connection %s created on demand (total: %d/%d)", newConn.ID, actualCount, maxConnections)
				}
				return newConn.DB, nil
			}
		} else {
			c.mu.Unlock()
		}
	}

	if c.Verbose {
		log.Printf("Pool at maximum limit (%d/%d), waiting for available PostgreSQL connection...", c.connections.Len(), maxConnections)
	}

	deadline, hasDeadline := ctx.Deadline()
	if hasDeadline && time.Until(deadline) < 50*time.Millisecond {
		return nil, fmt.Errorf("pool full (%d/%d PostgreSQL connections) and timeout too short", c.connections.Len(), maxConnections)
	}

	retryTicker := time.NewTicker(50 * time.Millisecond)
	defer retryTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context canceled while waiting for PostgreSQL connection (pool: %d/%d): %w", c.connections.Len(), maxConnections, ctx.Err())

		case <-retryTicker.C:
			orderedPairs := c.connections.GetOrderedV2()
			for _, pair := range orderedPairs {
				conn := pair.Value
				if !conn.InUse {
					err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledPostgresConnection) error {
						if (*v).InUse {
							return errors.New("PostgreSQL connection already in use")
						}

						if !c.IsConnectionValid(*v) {
							return errors.New("invalid PostgreSQL connection")
						}

						(*v).InUse = true
						(*v).LastUsed = time.Now()
						c.availableCount--
						return nil
					})

					if err == nil {
						if c.Verbose {
							log.Printf("PostgreSQL Connection %s obtained after waiting", conn.ID)
						}
						return conn.DB, nil
					}
				}
			}
		}
	}
}

func (c *PostgresConnectionInstance) ReleaseConnection(db *sqlx.DB) error {
	if db == nil {
		return errors.New("null PostgreSQL connection")
	}

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.DB == db {
			err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledPostgresConnection) error {
				if !(*v).InUse {
					return errors.New("PostgreSQL connection was not in use")
				}
				(*v).InUse = false
				(*v).LastUsed = time.Now()
				c.availableCount++
				return nil
			})

			if err != nil {
				return fmt.Errorf("error releasing PostgreSQL connection %s: %w", conn.ID, err)
			}

			if c.Verbose {
				log.Printf("PostgreSQL Connection %s released", conn.ID)
			}
			return nil
		}
	}

	return errors.New("PostgreSQL connection not found in pool")
}

func (c *PostgresConnectionInstance) Close() {
	log.Println("Closing PostgreSQL connection pool...")

	close(c.stopCh)
	c.wg.Wait()

	orderedPairs := c.connections.GetOrderedV2()
	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.DB != nil {
			conn.DB.Close()
			if c.Verbose {
				log.Printf("PostgreSQL Connection %s closed", conn.ID)
			}
		}
	}

	for _, pair := range orderedPairs {
		c.connections.Delete(pair.Key)
	}

	if c.Verbose {
		log.Println("PostgreSQL connection pool closed")
	}
}

func (c *PostgresConnectionInstance) GetPoolStats() map[string]interface{} {
	stats := make(map[string]interface{})

	total := c.connections.Len()
	available := 0
	inUse := 0

	orderedPairs := c.connections.GetOrderedV2()
	for _, pair := range orderedPairs {
		if pair.Value.InUse {
			inUse++
		} else {
			available++
		}
	}

	stats["total"] = total
	stats["available"] = available
	stats["in_use"] = inUse
	stats["min_connections"] = c.PgMinConnections
	stats["max_connections"] = c.PgMaxConnections
	stats["uptime_seconds"] = time.Since(c.created).Seconds()

	connections := make([]map[string]interface{}, 0)
	for _, pair := range orderedPairs {
		conn := pair.Value
		connInfo := map[string]interface{}{
			"id":           conn.ID,
			"in_use":       conn.InUse,
			"created_at":   conn.CreatedAt,
			"last_used":    conn.LastUsed,
			"age_seconds":  time.Since(conn.CreatedAt).Seconds(),
			"idle_seconds": time.Since(conn.LastUsed).Seconds(),
		}
		connections = append(connections, connInfo)
	}
	stats["connections"] = connections

	return stats
}

func (c *PostgresConnectionInstance) TriggerExpireConnections() {
	select {
	case c.expireConnectionsCh <- true:
		log.Println("Expire PostgreSQL connections requested")
	default:
		log.Println("Expire PostgreSQL connections request already in progress")
	}
}

func (c *PostgresConnectionInstance) TriggerRenewConnections() {
	select {
	case c.renewConnectionsCh <- true:
		log.Println("Renew PostgreSQL connections requested")
	default:
		log.Println("Renew PostgreSQL connections request already in progress")
	}
}

func (c *PostgresConnectionInstance) ResetPool() error {
	log.Println("Resetting PostgreSQL connection pool...")

	orderedPairs := c.connections.GetOrderedV2()
	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.DB != nil {
			conn.DB.Close()
		}
		c.connections.Delete(pair.Key)
	}

	for i := 0; i < c.PgMinConnections; i++ {
		conn, err := c.createConnection()
		if err != nil {
			if c.Verbose {
				log.Printf("Error recreating PostgreSQL connection %d: %v", i, err)
			}
			continue
		}
		c.connections.Set(conn.ID, conn)
	}

	c.availableCount = int32(c.PgMinConnections)
	if c.Verbose {
		log.Printf("PostgreSQL pool reset with %d connections", c.connections.Len())
	}
	return nil
}