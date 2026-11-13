import os
import getpass

print("Username:", getpass.getuser())
print("UID:", os.getuid())
print("GID:", os.getgid())
print("Groups:", os.getgroups())