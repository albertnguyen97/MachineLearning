import re

str = "My phone number is +84 812562496, my girl friend's one is +84 986668253"

result = re.findall("\+?\d+\ ?\d+", str)
print(result)

text = "Sample text with email addresses: john.doe@example-f.com and jane_smith12345@gmail.com"

# Define the regex pattern
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

# Find all email addresses in the text
email_addresses = re.findall(pattern, text)

# Print the list of email addresses
for email in email_addresses:
    print(email)