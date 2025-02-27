from datetime import date
import math 

#spędziłem około 25 minut

def calcDays(YYYY,MM,DD):
    today = date.today()
    birthday = date(YYYY, MM, DD)
    days = (today - birthday).days

    return days


def fizycznaFala(t):
    return math.sin(2 * math.pi / 23 * t)

def emocjonalnaFala(t):
    return math.sin(2 * math.pi / 28 * t)

def intelektualnaFala(t ):
    return math.sin(2 * math.pi / 33 * t)

def main():
    urodziny = input("Podaj date urodzin w formacie YYYY-MM-DD: ")
    urodziny = urodziny.split("-")
    urodziny = [int(i) for i in urodziny]
    t = calcDays(urodziny[0], urodziny[1], urodziny[2])
    print(f"Przeżyłeś już {t} dni")

    dzis = date.today()
    jutro = dzis.replace(day=dzis.day + 1)
    
    
    fizycznaFalaWynik = fizycznaFala(t)
    print(f"Twoja fizyczna fala to {fizycznaFalaWynik}" )
    if(fizycznaFalaWynik > 0.5):
        print("Dziś będzie siła")
    elif (fizycznaFalaWynik < -0.5):
        print("Poświęć dzisiejszy dzien na odpoczynek")

    fizycznaFalaJutro = fizycznaFala(t + 1)
    if (fizycznaFalaJutro > fizycznaFalaWynik):
        print("Nie martw się jutro będzie lepiej")
    

    emocjonalnaFalaWynik = emocjonalnaFala(t)
    print(f"Twoja emocjonalna fala to {emocjonalnaFalaWynik}")
    if (emocjonalnaFalaWynik > 0.5):
        print("Dziś dobra psycha")
    elif (emocjonalnaFalaWynik < -0.5):
        print("Dziś nie jest dobry dzień na podejmowanie ważnych decyzji")

    emocjonalnaFalaJutro = emocjonalnaFala(t + 1)
    if (emocjonalnaFalaJutro > emocjonalnaFalaWynik):
        print("Nie martw się jutro będzie lepiej")

    
    intelektualnaFalaWynik = intelektualnaFala(t)
    print(f"Twoja intelektualna fala to {intelektualnaFalaWynik}")
    if (intelektualnaFalaWynik > 0.5):
        print("Dziś jest dobry dzień na naukę")
    elif (intelektualnaFalaWynik < -0.5):
        print("Dziś nie jest dobry dzień na naukę")

    intelektualnaFalaJutro = intelektualnaFala(t + 1)
    if (intelektualnaFalaJutro > intelektualnaFalaWynik):
        print("Nie martw się jutro będzie lepiej")

    


main()
    