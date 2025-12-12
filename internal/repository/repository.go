package repository

import (
	"github.com/Dsouza10082/orus/internal/shared"
)

type Repository struct {
	PostgresPool shared.PostgresPool
	MilvusPool   shared.MilvusPool
}

func NewRepository() *Repository {
	return &Repository{
		PostgresPool: shared.GetPostgresConnectionInstance(),
		MilvusPool:   shared.GetMilvusConnectionInstance(),
	}
}

func (r *Repository) Close() {
	r.PostgresPool.Close()
	r.MilvusPool.Close()
}

func (r *Repository) GetPostgresPool() shared.PostgresPool {
	return r.PostgresPool
}