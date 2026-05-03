class BankAccount:

    bank_name = "NOTHDFC"

    def __init__(self, owner, balance=0):

        self.owner = owner
        self.balance = balance 

    def deposit(self, amount):

        self.balance += amount
        return self.balance
    
    def __repr__(self):
        return f"BankAccount(owner='{self.owner}', balance={self.balance})"

