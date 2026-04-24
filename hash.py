import streamlit_authenticator as stauth

# The password you want to log in with
my_password = 'mansi123'

# The NEW way to generate a hash in the updated library
hashed_password = stauth.Hasher.hash(my_password)

print("\n=========================================")
print("✅ Copy the scrambled text below:")
print(hashed_password)
print("=========================================\n")