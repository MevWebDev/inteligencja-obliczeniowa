import math
import datetime

#zajeło 30 sekund

def calculate_biorhythm(days, cycle):
    return math.sin(2 * math.pi * days / cycle)

def main():
    # Pobranie danych od użytkownika
    name = input("Podaj swoje imię: ")
    year = int(input("Podaj rok urodzenia: "))
    month = int(input("Podaj miesiąc urodzenia: "))
    day = int(input("Podaj dzień urodzenia: "))
    
    birth_date = datetime.date(year, month, day)
    today = datetime.date.today()
    days_lived = (today - birth_date).days
    
    # Obliczanie biorytmów
    physical = calculate_biorhythm(days_lived, 23)
    emotional = calculate_biorhythm(days_lived, 28)
    intellectual = calculate_biorhythm(days_lived, 33)
    
    print(f"\nCześć, {name}! Dziś jest {today}, to Twój {days_lived} dzień życia.")
    print(f"Biorytm fizyczny: {physical:.2f}")
    print(f"Biorytm emocjonalny: {emotional:.2f}")
    print(f"Biorytm intelektualny: {intellectual:.2f}")
    
    # Sprawdzenie prognozy na jutro
    tomorrow_lived = days_lived + 1
    physical_tomorrow = calculate_biorhythm(tomorrow_lived, 23)
    emotional_tomorrow = calculate_biorhythm(tomorrow_lived, 28)
    intellectual_tomorrow = calculate_biorhythm(tomorrow_lived, 33)
    
    def check_biorhythm(value, value_tomorrow, type_name):
        if value > 0.5:
            print(f"Twój {type_name} biorytm jest dzisiaj wysoki! Gratulacje!")
        elif value < -0.5:
            print(f"Twój {type_name} biorytm jest dzisiaj niski. Trzymaj się!")
            if value_tomorrow > value:
                print("Nie martw się. Jutro będzie lepiej!")
    
    check_biorhythm(physical, physical_tomorrow, "fizyczny")
    check_biorhythm(emotional, emotional_tomorrow, "emocjonalny")
    check_biorhythm(intellectual, intellectual_tomorrow, "intelektualny")
    
if __name__ == "__main__":
    main()
