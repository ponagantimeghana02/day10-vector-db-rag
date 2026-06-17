# access_control.py

from config import USER_ROLES


# ==================================
# Demo User Database
# ==================================

USERS = {

    "admin": {
        "password": "admin123",
        "role": "admin"
    },

    "employee": {
        "password": "employee123",
        "role": "employee"
    },

    "guest": {
        "password": "guest123",
        "role": "guest"
    }
}


# ==================================
# Authenticate User
# ==================================

def authenticate_user(
    username,
    password
):

    user = USERS.get(username)

    if not user:

        return None

    if user["password"] != password:

        return None

    return user


# ==================================
# Get User Role
# ==================================

def get_user_role(user):

    return user["role"]


# ==================================
# Get Access Level
# ==================================

def get_access_level(role):

    role_info = USER_ROLES.get(role)

    if not role_info:

        return "public"

    return role_info["access"]


# ==================================
# Metadata Filter
# ==================================

def get_metadata_filter(role):

    access_level = get_access_level(role)

    if access_level == "all":

        return None

    return {

        "access_level": access_level

    }


# ==================================
# Permission Check
# ==================================

def has_access(
    role,
    document_access_level
):

    if role == "admin":

        return True

    role_access = get_access_level(role)

    return (
        role_access ==
        document_access_level
    )


# ==================================
# Login
# ==================================

def login():

    print("\nLogin")

    print("-" * 50)

    username = input(
        "Username: "
    )

    password = input(
        "Password: "
    )

    user = authenticate_user(
        username,
        password
    )

    if not user:

        print(
            "\nInvalid credentials."
        )

        return None

    print(
        f"\nWelcome {username}"
    )

    print(
        f"Role: {user['role']}"
    )

    return user


# ==================================
# Show Available Roles
# ==================================

def show_roles():

    print("\nRoles")

    print("-" * 50)

    for role in USER_ROLES:

        print(role)


# ==================================
# Test
# ==================================

if __name__ == "__main__":

    user = login()

    if user:

        role = get_user_role(user)

        print(
            "\nMetadata Filter:"
        )

        print(
            get_metadata_filter(role)
        )