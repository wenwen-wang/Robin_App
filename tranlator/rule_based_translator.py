from mistralai import Mistral
import os


# === CONFIGURATION ===
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

def translate_de_to_en(text,MISTRAL_API_KEY):
    # === MISTRAL CLIENT ===
    user_prompt = text
    system_prompt = "Please translate the text to English."


    # === TRANSLATE ===
    client = Mistral(MISTRAL_API_KEY)
    response =client.chat.complete(
        model="mistral-small-latest",
        temperature=0.0,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
    )
    # === PARSE RESPONSE ===
    translation = response.choices[0].message.content.strip()
    # === OUTPUT ===
    return translation


input_text = "Die Initiierung eines öffentlichen Bücherschranks im Stadtteil soll die Lesekultur fördern und den Zugang " \
             "zu Literatur für alle Bewohner erleichtern. Der Bücherschrank wird als zentraler Treffpunkt dienen, an dem " \
             "Bücher kostenlos getauscht und entliehen werden können. Dieses Projekt stärkt nicht nur den sozialen Zusammenhalt " \
             "im Stadtteil, sondern fördert auch den nachhaltigen Umgang mit Ressourcen, indem Bücher weitergegeben statt entsorgt werden. " \
             "Von diesem Projekt profitieren insbesondere Familien, Kinder und Jugendliche, die durch den einfachen Zugang zu Büchern " \
             "ihre Lesefähigkeiten verbessern können. Lokale Schulen und Bildungseinrichtungen können den Bücherschrank in ihre Programme integrieren, " \
             "um das Interesse am Lesen zu fördern. Der Bücherschrank bietet zudem eine Plattform für lokale Autoren, ihre Werke einer breiteren Öffentlichkeit " \
             "zugänglich zu machen. Für die Stadt entsteht ein Mehrwert durch die Förderung der Bildung und die Stärkung des Gemeinschaftsgefühls im Stadtteil. " \
             "Der Bücherschrank kann als Vorzeigeprojekt für weitere Initiativen zur Förderung der Lesekultur dienen. Die geschätzten Kosten für das Projekt belaufen sich auf ca. 2.000 Euro. Ich bin eine Privatperson."
#print(translate_de_to_en(input_text, MISTRAL_API_KEY))

