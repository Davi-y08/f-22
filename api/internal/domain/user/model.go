package user

import (
	"time"

	"github.com/google/uuid"
)

type User struct {
	ID				uuid.UUID	`gorm:"type:uuid;default:gen_random_uuid();primaryKey"`
	Name 			string 		`json:"name"`
	Email 			string 		`json:"email" gorm:"uniqueIndex;size:180"`
	PasswordHash 	string 		`json:"-"`
	Role 			string 		`json:"role" gorm:"default:user"`
	CreatedAt 		time.Time
	UpdatedAt 		time.Time
}