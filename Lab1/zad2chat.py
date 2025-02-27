import math
import random
import matplotlib.pyplot as plt

# Stałe
v0 = 50  # m/s
h = 100  # m
g = 9.81  # m/s²

# Losowanie celu
cel = random.randint(50, 340)
print(f"Cel znajduje się w odległości {cel} metrów.")

proby = 0

while True:
    try:
        alpha = float(input("Podaj kąt strzału w stopniach: "))
    except ValueError:
        print("To nie jest prawidłowa liczba. Spróbuj ponownie.")
        continue

    proby += 1

    # Konwersja stopni na radiany
    alpha_rad = math.radians(alpha)

    # Obliczanie składowych prędkości
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    # Obliczanie odległości
    czesc_pierwiastkowa = math.sqrt((v0 * sin_alpha)**2 + 2 * g * h)
    d = (v0 * cos_alpha / g) * (v0 * sin_alpha + czesc_pierwiastkowa)

    print(f"Pocisk przeleciał {d:.2f} metrów.")

    # Sprawdzenie trafienia
    if cel - 5 <= d <= cel + 5:
        print(f"CEL TRAFIONY! Liczba prób: {proby}")
        
        # Rysowanie trajektorii
        t_total = (v0 * sin_alpha + math.sqrt((v0 * sin_alpha)**2 + 2 * g * h)) / g
        czas = [t_total * i / 100 for i in range(101)]  # 100 punktów czasowych
        x = [v0 * cos_alpha * t for t in czas]
        y = [h + v0 * sin_alpha * t - 0.5 * g * t**2 for t in czas]
        
        plt.figure()
        plt.plot(x, y, 'b-')  # Niebieska linia ciągła
        plt.grid(True)
        plt.xlabel("Distance (m)")
        plt.ylabel("Height (m)")
        plt.title("Trajektoria pocisku Warwolf")
        plt.savefig("trajektoria.png")
        plt.close()
        
        break
    else:
        print("Pudło! Spróbuj ponownie.")