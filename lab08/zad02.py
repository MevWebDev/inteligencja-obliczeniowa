import text2emotion as te
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
negative_review = """
I am writing this review with great disappointment after what was supposed to be a relaxing weekend getaway turned into an absolute nightmare. This hotel has completely failed to meet even the most basic standards of hospitality and customer service.

From the moment we arrived, everything went wrong. The check-in process took over 45 minutes because their computer system was "experiencing technical difficulties." The front desk staff seemed completely overwhelmed and offered no apology or explanation for the delay. When we finally got to our room on the 8th floor, we discovered that the key cards didn't work. We had to make three trips back to the front desk to get new cards programmed.

The room itself was a disaster. Despite booking a "deluxe ocean view" room, we were given a cramped space overlooking the parking garage. The carpet was stained and smelled musty, as if it hadn't been properly cleaned in months. The bathroom was even worse - the toilet was clogged, the shower had no hot water, and there were hair and soap scum all over the tiles. The towels looked like they hadn't been changed from the previous guests.

The air conditioning unit was broken and making horrible grinding noises all night. When we called maintenance, they said they couldn't fix it until the next day. The bed was lumpy and uncomfortable, with sheets that felt dirty and pillows that were flat as pancakes. We barely got any sleep because of the noise from the broken AC and the party happening in the room next door until 3 AM.

The hotel amenities were equally disappointing. The pool was closed for "maintenance" with no prior notice, despite this being advertised as one of their main features. The fitness center had broken equipment and smelled terrible. The restaurant was overpriced with terrible food - I ordered a simple chicken sandwich that came out cold and dry, clearly sitting under heat lamps for hours.

The worst part was the staff's attitude. When we complained about the numerous problems, the manager was dismissive and rude. He basically told us that we were "lucky to have a room at all" during peak season and that there was nothing he could do about our complaints. No offer of compensation, no attempt to relocate us, no genuine concern for our experience.

The final straw came when we discovered that housekeeping had entered our room while we were out and somehow managed to break our laptop screen. When we reported this to the front desk, they denied any responsibility and suggested we must have damaged it ourselves. The security footage was conveniently "unavailable" for review.

To make matters worse, when we checked out, they tried to charge us extra fees for "excessive towel usage" and "damage to hotel property" - referring to the broken laptop that their staff had damaged! It took nearly an hour of arguing to get these fraudulent charges removed.

I have stayed in hundreds of hotels around the world, and this was by far the worst experience I have ever had. The combination of dirty, broken facilities, terrible food, incompetent staff, and dishonest management makes this place completely unacceptable. Save your money and book literally anywhere else. This hotel is a complete scam and I will be reporting them to the tourism board and posting warnings on every travel website I can find. Absolutely disgusting establishment that should be shut down immediately.
"""

positive_review = """
I cannot express enough how absolutely wonderful our stay at this magnificent hotel was! From the moment we walked through the elegant lobby doors, we were treated like royalty by the most professional and warm staff I have ever encountered in my travels around the world.

The check-in process was seamless and efficient. The front desk manager, Sarah, personally welcomed us with genuine enthusiasm and took the time to explain all the hotel amenities and local attractions. She even arranged for a complimentary room upgrade to a stunning suite with breathtaking panoramic views of the ocean. The attention to detail was extraordinary - they had personalized welcome gifts waiting in our room, including local chocolates and a bottle of champagne to celebrate our anniversary.

Our suite was absolutely spectacular! Spacious, immaculately clean, and beautifully decorated with elegant furnishings and artwork. The bed was like sleeping on a cloud - the most comfortable mattress and pillows I've ever experienced. The bathroom was a luxury spa experience with a deep soaking tub, rainfall shower, heated floors, and premium toiletries. Every surface sparkled, and the housekeeping attention to detail was remarkable - they even folded our clothes and arranged our personal items thoughtfully.

The hotel amenities exceeded all expectations. The infinity pool overlooking the ocean was absolutely stunning, with comfortable loungers and attentive pool service staff who brought us refreshing drinks and snacks throughout the day. The fitness center was state-of-the-art with brand new equipment, and the spa was pure heaven - I had the most relaxing massage of my life with incredible ocean views from the treatment room.

The dining experiences were phenomenal! The main restaurant offered exquisite cuisine with fresh, locally-sourced ingredients prepared by obviously talented chefs. Every meal was a culinary masterpiece beautifully presented. The breakfast buffet was extensive with everything from fresh tropical fruits to made-to-order omelets and gourmet pastries. The rooftop bar became our favorite evening spot, with creative cocktails, live music, and the most romantic sunset views imaginable.

What truly set this hotel apart was the exceptional service from every single staff member. The concierge team went above and beyond to arrange special experiences for us, including a private sunset cruise and reservations at exclusive local restaurants. The housekeeping staff were incredibly thorough and friendly, always greeting us with warm smiles. The restaurant servers were knowledgeable about the menu and wine pairings, making excellent recommendations that perfectly suited our tastes.

The location is absolutely perfect - right on the beautiful white sand beach with crystal clear waters, yet close enough to the city center for easy exploration. The hotel provided complimentary shuttle service to local attractions and shopping areas, which was incredibly convenient.

Special touches throughout our stay showed how much this hotel cares about their guests' experience. They surprised us with a romantic dinner setup on our private balcony for our anniversary, complete with rose petals, candles, and a specially prepared menu. When they learned it was our first visit to the area, they arranged a personalized city tour with a knowledgeable local guide.

The value for money is outstanding considering the luxury level and exceptional service provided. Every penny spent was worth it for the memories we created and the level of pampering we received. This hotel has set a new standard for excellence in hospitality.

We have already booked our return visit for next year and have recommended this incredible hotel to all our friends and family. This is exactly what a luxury hotel experience should be - flawless service, beautiful accommodations, amazing amenities, and staff who genuinely care about making your stay unforgettable. This hotel has earned a permanent place on our list of favorite destinations, and we cannot wait to return to this paradise! Five stars is not enough - this place deserves ten stars!
"""

# Pobranie zasobów VADER (wykonaj raz)
nltk.download('vader_lexicon')

# Inicjalizacja analizatora
sia = SentimentIntensityAnalyzer()


def analyze_with_vader(text):
    scores = sia.polarity_scores(text)
    print(f"VADER Scores: {scores}")
    return scores


# pip install emoji==1.7.0
# Analiza opinii
print("--- Pozytywna opinia ---")
vader_pos = analyze_with_vader(positive_review)

print("\n--- Negatywna opinia ---")
vader_neg = analyze_with_vader(negative_review)


print("text-to-emotion")
print("--- Negatywna ---")
a = te.get_emotion(negative_review)
print(a)
print("--- pozytywna ---")
b = te.get_emotion(positive_review)
print(b)

#wyniki nie sa zgodne z oczekiwaniami
