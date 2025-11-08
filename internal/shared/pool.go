package shared

import (
	"context"
	"sync"
	"time"

	concurrentmap "github.com/Dsouza10082/ConcurrentOrderedMap"
	"github.com/jmoiron/sqlx"
	"github.com/milvus-io/milvus-sdk-go/v2/client"
)

type PooledPostgresConnection struct {
	DB        *sqlx.DB
	CreatedAt time.Time
	LastUsed  time.Time
	ID        string
	InUse     bool
}

type PooledMilvusConnection struct {
	Client    client.Client
	CreatedAt time.Time
	LastUsed  time.Time
	ID        string
	InUse     bool
}

type PostgresConnectionInstance struct {
	mu                  sync.RWMutex
	created             time.Time
	connections         *concurrentmap.ConcurrentOrderedMap[string, *PooledPostgresConnection]
	availableCount      int32
	PgDBHost            string
	PgMinConnections    int
	PgMaxConnections    int
	PgConnectionLifeTime int
	expireConnectionsCh chan bool
	renewConnectionsCh  chan bool
	stopCh              chan bool
	wg                  sync.WaitGroup
	connectionCounter   int
	Verbose             bool
}

type MilvusConnectionInstance struct {
	mu                       sync.RWMutex
	created                  time.Time
	connections              *concurrentmap.ConcurrentOrderedMap[string, *PooledMilvusConnection]
	availableCount           int32
	MilvusHost               string
	MilvusUser               string
	MilvusPassword           string
	MilvusDatabase           string
	MilvusMaxConnections     int
	MilvusMinConnections     int
	MilvusConnectionLifeTime int
	expireConnectionsCh      chan bool
	renewConnectionsCh       chan bool
	stopCh                   chan bool
	wg                       sync.WaitGroup
	connectionCounter        int
	Verbose                  bool
}

var (
	postgresInstance *PostgresConnectionInstance
	postgresOnce     sync.Once
	milvusInstance   *MilvusConnectionInstance
	milvusOnce       sync.Once
)

type PostgresPool interface {
	GetConnection(ctx context.Context) (*sqlx.DB, error)
	ReleaseConnection(db *sqlx.DB) error
	Close()
	GetPoolStats() map[string]interface{}
	TriggerExpireConnections()
	TriggerRenewConnections()
	ResetPool() error
}

type MilvusPool interface {
	GetConnection(ctx context.Context) (client.Client, error)
	ReleaseConnection(c client.Client) error
	Close()
	GetPoolStats() map[string]interface{}
	TriggerExpireConnections()
	TriggerRenewConnections()
	ResetPool() error
}