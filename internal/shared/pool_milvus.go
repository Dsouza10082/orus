package shared

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	concurrentmap "github.com/Dsouza10082/ConcurrentOrderedMap"
	"github.com/Dsouza10082/orus/config"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

func GetMilvusConnectionInstance() *MilvusConnectionInstance {
	milvusOnce.Do(func() {
		milvusInstance = newMilvusConnectionInstance()
	})
	return milvusInstance
}

func newMilvusConnectionInstance() *MilvusConnectionInstance {
	params := config.GetParameters()
	connInst := &MilvusConnectionInstance{
		created:                  time.Now(),
		connections:              concurrentmap.NewConcurrentOrderedMapWithCapacity[string, *PooledMilvusConnection](params.MilvusMaxConnections),
		availableCount:           0,
		MilvusHost:               params.MilvusHost,
		MilvusUser:               params.MilvusUser,
		MilvusPassword:           params.MilvusPassword,
		MilvusDatabase:           params.MilvusDatabase,
		MilvusMaxConnections:     params.MilvusMaxConnections,
		MilvusMinConnections:     params.MilvusMinConnections,
		MilvusConnectionLifeTime: params.MilvusConnectionLifeTime,
		expireConnectionsCh:      make(chan bool, 1),
		renewConnectionsCh:       make(chan bool, 1),
		stopCh:                   make(chan bool),
		connectionCounter:        0,
		Verbose:                  false,
	}

	for i := 0; i < params.MilvusMinConnections; i++ {
		conn, err := connInst.createConnection()
		if err != nil {
			log.Printf("Error creating initial Milvus connection %d: %v", i, err)
			continue
		}
		connInst.connections.Set(conn.ID, conn)
		connInst.availableCount++
	}

	connInst.wg.Add(1)
	go connInst.maintainPool()

	log.Printf("Milvus connection pool initialized with %d connections", connInst.connections.Len())
	return connInst
}

func (c *MilvusConnectionInstance) createConnection() (*PooledMilvusConnection, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Criar cliente Milvus com as configurações
	milvusClient, err := client.NewClient(ctx, client.Config{
		Address:  c.MilvusHost,
		Username: c.MilvusUser,
		Password: c.MilvusPassword,
		DBName:   c.MilvusDatabase,
	})
	if err != nil {
		return nil, fmt.Errorf("error creating Milvus client: %w", err)
	}

	// Verificar se a conexão está funcionando
	if err := c.pingMilvusConnection(ctx, milvusClient); err != nil {
		milvusClient.Close()
		return nil, fmt.Errorf("error pinging Milvus connection: %w", err)
	}

	c.mu.Lock()
	c.connectionCounter++
	id := fmt.Sprintf("milvus_conn_%d_%d", c.connectionCounter, time.Now().UnixNano())
	c.mu.Unlock()

	return &PooledMilvusConnection{
		Client:    milvusClient,
		CreatedAt: time.Now(),
		LastUsed:  time.Now(),
		ID:        id,
		InUse:     false,
	}, nil
}

func (c *MilvusConnectionInstance) pingMilvusConnection(ctx context.Context, milvusClient client.Client) error {
	// Tentar listar databases como forma de verificar a conexão
	_, err := milvusClient.ListDatabases(ctx)
	return err
}

func (c *MilvusConnectionInstance) maintainPool() {
	defer c.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			log.Println("Stopping pool maintenance for Milvus")
			return

		case <-ticker.C:
			c.performMaintenance()

		case <-c.expireConnectionsCh:
			log.Println("Request to expire old connections for Milvus")
			c.expireOldConnections()
			c.ensureMinimumConnections()

		case <-c.renewConnectionsCh:
			log.Println("Request to renew connections for Milvus")
			c.ensureMaximumConnections()
		}
	}
}

func (c *MilvusConnectionInstance) performMaintenance() {
	expiredCount := 0
	validCount := 0
	inUseCount := 0
	availableCount := 0

	orderedPairs := c.connections.GetOrderedV2()
	toRemove := make([]string, 0)

	for _, pair := range orderedPairs {
		conn := pair.Value

		if !c.IsConnectionValid(conn) {
			conn.Client.Close()
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
		log.Printf("Milvus Pool Status - Total: %d, Valid: %d, Available: %d, In use: %d, Expired: %d",
			c.connections.Len(), validCount, availableCount, inUseCount, expiredCount)
	}

	if validCount < c.MilvusMinConnections {
		go c.ensureMinimumConnections()
	}
}

func (c *MilvusConnectionInstance) expireOldConnections() {
	connectionLifeTime := time.Duration(c.MilvusConnectionLifeTime) * time.Second
	toRemove := make([]string, 0)

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value
		if time.Since(conn.CreatedAt) > connectionLifeTime {
			conn.Client.Close()
			toRemove = append(toRemove, pair.Key)
			if c.Verbose {
				log.Printf("Milvus Connection %s forcibly expired", conn.ID)
			}
		}
	}

	for _, id := range toRemove {
		c.connections.Delete(id)
	}
}

func (c *MilvusConnectionInstance) ensureMinimumConnections() {
	currentCount := c.connections.Len()
	needed := c.MilvusMinConnections - currentCount

	if needed <= 0 {
		return
	}

	if c.Verbose {
		log.Printf("Creating %d Milvus connections to reach minimum", needed)
	}

	for i := 0; i < needed; i++ {
		conn, err := c.createConnection()
		if err != nil {
			if c.Verbose {
				log.Printf("Error creating replacement Milvus connection: %v", err)
			}
			continue
		}

		c.connections.Set(conn.ID, conn)
		c.availableCount++
		if c.Verbose {
			log.Printf("New Milvus connection %s added to pool", conn.ID)
		}
	}
}

func (c *MilvusConnectionInstance) ensureMaximumConnections() {
	c.mu.Lock()
	defer c.mu.Unlock()

	currentCount := c.connections.Len()
	needed := c.MilvusMaxConnections - currentCount

	if needed <= 0 {
		if c.Verbose {
			log.Printf("Pool already at maximum (%d Milvus connections)", currentCount)
		}
		return
	}

	if c.Verbose {
		log.Printf("Creating %d additional Milvus connections", needed)
	}

	for i := 0; i < needed; i++ {
		currentCount = c.connections.Len()
		if currentCount >= c.MilvusMaxConnections {
			if c.Verbose {
				log.Printf("Reached maximum Milvus connections during creation")
			}
			break
		}

		c.mu.Unlock()
		conn, err := c.createConnection()
		c.mu.Lock()

		if err != nil {
			log.Printf("Error creating additional Milvus connection: %v", err)
			continue
		}

		currentCount = c.connections.Len()
		if currentCount >= c.MilvusMaxConnections {
			conn.Client.Close()
			if c.Verbose {
				log.Printf("Maximum Milvus connections reached, closing excess connection %s", conn.ID)
			}
			break
		}

		c.connections.Set(conn.ID, conn)
		c.availableCount++
		if c.Verbose {
			log.Printf("Additional Milvus connection %s created", conn.ID)
		}
	}
}

func (c *MilvusConnectionInstance) IsConnectionValid(conn *PooledMilvusConnection) bool {
	if conn == nil || conn.Client == nil {
		return false
	}

	connectionLifeTime := time.Duration(c.MilvusConnectionLifeTime) * time.Second
	if time.Since(conn.CreatedAt) > connectionLifeTime {
		if c.Verbose {
			log.Printf("Milvus Connection %s expired due to lifetime", conn.ID)
		}
		return false
	}

	maxIdleTime := 5 * time.Minute
	if !conn.InUse && time.Since(conn.LastUsed) > maxIdleTime {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()

		if err := c.pingMilvusConnection(ctx, conn.Client); err != nil {
			if c.Verbose {
				log.Printf("Milvus Connection %s failed ping: %v", conn.ID, err)
			}
			return false
		}
	}

	return true
}

func (c *MilvusConnectionInstance) GetConnection(ctx context.Context) (client.Client, error) {
	c.mu.RLock()
	maxConnections := c.MilvusMaxConnections
	c.mu.RUnlock()

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value

		if !conn.InUse {
			err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledMilvusConnection) error {
				if (*v).InUse {
					return errors.New("milvus connection already in use")
				}

				if !c.IsConnectionValid(*v) {
					return errors.New("invalid Milvus connection")
				}

				(*v).InUse = true
				(*v).LastUsed = time.Now()
				c.availableCount--
				return nil
			})

			if err == nil {
				if c.Verbose {
					log.Printf("Milvus Connection %s obtained from pool", conn.ID)
				}
				return conn.Client, nil
			}
		}
	}

	currentCount := c.connections.Len()
	if currentCount < maxConnections {
		c.mu.Lock()
		currentCount = c.connections.Len()
		if currentCount < c.MilvusMaxConnections {
			c.mu.Unlock()
			newConn, err := c.createConnection()
			c.mu.Lock()

			if err != nil {
				c.mu.Unlock()
				return nil, fmt.Errorf("pool exhausted and could not create new Milvus connection: %w", err)
			}

			currentCount = c.connections.Len()
			if currentCount >= c.MilvusMaxConnections {
				c.mu.Unlock()
				newConn.Client.Close()
				if c.Verbose {
					log.Printf("Maximum Milvus connections reached while creating new connection, closing %s", newConn.ID)
				}
			} else {
				newConn.InUse = true
				newConn.LastUsed = time.Now()
				c.connections.Set(newConn.ID, newConn)
				actualCount := c.connections.Len()
				c.mu.Unlock()

				if c.Verbose {
					log.Printf("New Milvus connection %s created on demand (total: %d/%d)", newConn.ID, actualCount, maxConnections)
				}
				return newConn.Client, nil
			}
		} else {
			c.mu.Unlock()
		}
	}

	if c.Verbose {
		log.Printf("Pool at maximum limit (%d/%d), waiting for available Milvus connection...", c.connections.Len(), maxConnections)
	}

	deadline, hasDeadline := ctx.Deadline()
	if hasDeadline && time.Until(deadline) < 50*time.Millisecond {
		return nil, fmt.Errorf("pool full (%d/%d Milvus connections) and timeout too short", c.connections.Len(), maxConnections)
	}

	retryTicker := time.NewTicker(50 * time.Millisecond)
	defer retryTicker.Stop()

	for {
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("context canceled while waiting for Milvus connection (pool: %d/%d): %w", c.connections.Len(), maxConnections, ctx.Err())

		case <-retryTicker.C:
			orderedPairs := c.connections.GetOrderedV2()
			for _, pair := range orderedPairs {
				conn := pair.Value
				if !conn.InUse {
					err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledMilvusConnection) error {
						if (*v).InUse {
							return errors.New("milvus connection already in use")
						}

						if !c.IsConnectionValid(*v) {
							return errors.New("invalid Milvus connection")
						}

						(*v).InUse = true
						(*v).LastUsed = time.Now()
						c.availableCount--
						return nil
					})

					if err == nil {
						if c.Verbose {
							log.Printf("Milvus Connection %s obtained after waiting", conn.ID)
						}
						return conn.Client, nil
					}
				}
			}
		}
	}
}

func (c *MilvusConnectionInstance) ReleaseConnection(milvusClient client.Client) error {
	if milvusClient == nil {
		return errors.New("null Milvus connection")
	}

	orderedPairs := c.connections.GetOrderedV2()

	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.Client == milvusClient {
			err := c.connections.UpdateWithPointer(pair.Key, func(v **PooledMilvusConnection) error {
				if !(*v).InUse {
					return errors.New("nilvus connection was not in use")
				}
				(*v).InUse = false
				(*v).LastUsed = time.Now()
				c.availableCount++
				return nil
			})

			if err != nil {
				return fmt.Errorf("error releasing Milvus connection %s: %w", conn.ID, err)
			}

			if c.Verbose {
				log.Printf("Milvus Connection %s released", conn.ID)
			}
			return nil
		}
	}

	return errors.New("milvus connection not found in pool")
}

func (c *MilvusConnectionInstance) Close() {
	log.Println("Closing Milvus connection pool...")

	close(c.stopCh)
	c.wg.Wait()

	orderedPairs := c.connections.GetOrderedV2()
	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.Client != nil {
			conn.Client.Close()
			if c.Verbose {
				log.Printf("Milvus Connection %s closed", conn.ID)
			}
		}
	}

	for _, pair := range orderedPairs {
		c.connections.Delete(pair.Key)
	}

	if c.Verbose {
		log.Println("Milvus connection pool closed")
	}
}

func (c *MilvusConnectionInstance) GetPoolStats() map[string]interface{} {
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
	stats["min_connections"] = c.MilvusMinConnections
	stats["max_connections"] = c.MilvusMaxConnections
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

func (c *MilvusConnectionInstance) TriggerExpireConnections() {
	select {
	case c.expireConnectionsCh <- true:
		log.Println("Expire Milvus connections requested")
	default:
		log.Println("Expire Milvus connections request already in progress")
	}
}

func (c *MilvusConnectionInstance) TriggerRenewConnections() {
	select {
	case c.renewConnectionsCh <- true:
		log.Println("Renew Milvus connections requested")
	default:
		log.Println("Renew Milvus connections request already in progress")
	}
}

func (c *MilvusConnectionInstance) ResetPool() error {
	log.Println("Resetting Milvus connection pool...")

	orderedPairs := c.connections.GetOrderedV2()
	for _, pair := range orderedPairs {
		conn := pair.Value
		if conn.Client != nil {
			conn.Client.Close()
		}
		c.connections.Delete(pair.Key)
	}

	for i := 0; i < c.MilvusMinConnections; i++ {
		conn, err := c.createConnection()
		if err != nil {
			if c.Verbose {
				log.Printf("Error recreating Milvus connection %d: %v", i, err)
			}
			continue
		}
		c.connections.Set(conn.ID, conn)
	}

	c.availableCount = int32(c.MilvusMinConnections)
	if c.Verbose {
		log.Printf("Milvus pool reset with %d connections", c.connections.Len())
	}
	return nil
}