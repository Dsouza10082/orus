package repository

import (
	"context"
	_ "embed"

	"github.com/Dsouza10082/orus/internal/model"
)

//go:embed query/users/SELECT_ALL_USERS.sql
var SELECT_ALL_USERS string

type UsersRepository struct {
	*Repository
}

func NewUsersRepository() *UsersRepository {
	return &UsersRepository{
		Repository: NewRepository(),
	}
}

func (r *Repository) CreateUser(user *model.Users) error {
	return nil
}

func (r *Repository) GetUser(serial string) (*model.Users, error) {
	db, err := r.PostgresPool.GetConnection(context.Background())
	if err != nil {
		return nil, err
	}
	defer db.Close()

	query := "SELECT * FROM users WHERE serial = $1"
	var user model.Users
	err = db.Get(&user, query, serial)
	if err != nil {
		return nil, err
	}
	return nil, nil
}

func (r *Repository) UpdateUser(user *model.Users) error {
	return nil
}