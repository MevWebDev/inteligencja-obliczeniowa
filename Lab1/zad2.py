import math
import random
import matplotlib.pyplot as plt

def strzał(kąt):
    wysokość = 100
    prędkość = 50
    g = 9.81

    kąt = float(kąt)
    alpha_rad = math.radians(kąt)
    cos_alpha = math.cos(alpha_rad)
    sin_alpha = math.sin(alpha_rad)

    czesc_pierwiastkowa = math.sqrt((prędkość * sin_alpha)**2 + 2 * g * wysokość)
    d = (prędkość * cos_alpha / g) * (prędkość * sin_alpha + czesc_pierwiastkowa)

    print(f"Pocisk przeleciał {d:.2f} metrów.")
    
    return d, prędkość, cos_alpha, sin_alpha, wysokość, g

def main():
    cel = random.randint(50, 340)
    print(f"Cel: {cel}")
    
    while True:
        kąt = input("Podaj kąt strzału: ")
        d, prędkość, cos_alpha, sin_alpha, wysokość, g = strzał(kąt)
        
       

       
        t_total = (prędkość * sin_alpha + math.sqrt((prędkość * sin_alpha)**2 + 2 * g * wysokość)) / g
        
       
        czas = [t_total * i / 100 for i in range(101)]
        x = [prędkość * cos_alpha * t for t in czas]
        y = [wysokość + prędkość * sin_alpha * t - 0.5 * g * t**2 for t in czas]
        
        

        if abs(d - cel) <= 5:
            print("Trafiłeś w cel")
            plt.figure()
            plt.plot(x, y, 'b-')  
            plt.grid(True)
            plt.xlabel("Distance (m)")
            plt.ylabel("Height (m)")
            plt.title("Trajektoria pocisku Warwolf")
            plt.savefig("trajektoria.png")
            plt.close()
            print("Wykres zapisany jako trajektoria.png.")
            break
        else:
            print("Nie trafione")

        

if __name__ == '__main__':
    main()