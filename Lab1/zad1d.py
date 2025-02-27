from datetime import date, timedelta
import math

def calc_days(year, month, day):
    today = date.today()
    birthday = date(year, month, day)
    days_lived = (today - birthday).days
    return days_lived

def physical_wave(t):
    return math.sin(2 * math.pi / 23 * t)

def emotional_wave(t):
    return math.sin(2 * math.pi / 28 * t)

def intellectual_wave(t):
    return math.sin(2 * math.pi / 33 * t)

def analyze_wave(value, positive_message, negative_message):
    print(f"Wartość fali: {value:.2f}")
    if value > 0.5:
        print(positive_message)
    elif value < -0.5:
        print(negative_message)

def main():
    birth_date = input("Podaj datę urodzenia w formacie YYYY-MM-DD: ")
    try:
        year, month, day = map(int, birth_date.split("-"))
        days_lived = calc_days(year, month, day)
        print(f"Przeżyłeś już {days_lived} dni")
        
        today_physical = physical_wave(days_lived)
        print("\n--- Fala fizyczna ---")
        analyze_wave(today_physical, "Dziś będzie siła!", "Poświęć dzisiejszy dzień na odpoczynek.")
        
        tomorrow_physical = physical_wave(days_lived + 1)
        if tomorrow_physical > today_physical:
            print("Jutro będzie lepiej!")
        
        today_emotional = emotional_wave(days_lived)
        print("\n--- Fala emocjonalna ---")
        analyze_wave(today_emotional, "Dziś dobra psycha!", "Dziś nie jest dobry dzień na podejmowanie ważnych decyzji.")
        
        tomorrow_emotional = emotional_wave(days_lived + 1)
        if tomorrow_emotional > today_emotional:
            print("Jutro będzie lepiej!")
        
        today_intellectual = intellectual_wave(days_lived)
        print("\n--- Fala intelektualna ---")
        analyze_wave(today_intellectual, "Dziś jest dobry dzień na naukę!", "Dziś nie jest dobry dzień na naukę.")
        
        tomorrow_intellectual = intellectual_wave(days_lived + 1)
        if tomorrow_intellectual > today_intellectual:
            print("Jutro będzie lepiej!")
        
    except ValueError:
        print("Niepoprawny format daty. Użyj formatu YYYY-MM-DD.")

if __name__ == "__main__":
    main()
