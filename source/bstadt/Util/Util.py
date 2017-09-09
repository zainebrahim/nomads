def delta_epsilon(a, b, e):
    return abs(a-b) < e

if __name__ == '__main__':
    print(delta_epsilon(1., 1.1, .2) == True)
    print(delta_epsilon(1., 1.1, .01) == False)
