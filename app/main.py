from db import Base, engine
# # Create tables
# Base.metadata.create_all(engine)

# # Insert data
# Session = sessionmaker(bind=engine)
# session = Session()

# new_user = User(username='john_doe', email='john.doe@example.com')
# session.add(new_user)
# session.commit()

# # Query data
# users = session.query(User).all()
# for user in users:
#     print(f"User ID: {user.id}, Username: {user.username}, Email: {user.email}")

def main() -> None:
    Base.metadata.create_all(engine)

if __name__ == "__main__":
    main()