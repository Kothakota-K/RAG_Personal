from Bank import BankAccount

def main():
    acc1 = BankAccount("jose", 1000)
    # acc1.bank_name = "HDFC"
    acc2 = BankAccount("karthik")

    acc1.deposit(2000)

    print(acc1.bank_name)
    print(acc2.bank_name)
    print(acc1)
    print(acc2)

if __name__ == "__main__":
    main()