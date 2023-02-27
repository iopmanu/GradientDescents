from gd import BaseDescent

def main() -> None:
    d = BaseDescent(max_iter=5, dimension=3, learning_rate='constant')
    print(d.__dict__)


if __name__ == "__main__":
    main()