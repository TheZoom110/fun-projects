def flames(name1: str, name2: str) -> str:
    a = list(name1.upper())
    b = list(name2.upper())
    for i in list(a):
        if i in b:
            a.remove(i)
            b.remove(i)
    n = len(a) + len(b)

    f = "FLAMES"
    f_len = 6

    while f_len > 1:
        r = n % f_len
        if r == 0:
            f = f[: f_len - 1]
            f_len -= 1
        else:
            f = f[r:] + f[: r - 1]
            f_len -= 1

    return f[0]

n1 = input("Enter name 1: ")
n2 = input("Enter name 2: ")
rel = {'F': 'Friendship', 'L': 'Lovers', 'A': 'Affection', 'M': 'Marriage', 'E': 'Enemies', 'S': 'Siblings'}
print(f"Relation of {n1} with {n2} is {rel[flames(n1, n2)]}.")
