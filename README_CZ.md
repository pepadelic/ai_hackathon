# 🎬 Generátor filmových zápletek

Streamlit aplikace pro generování originálních filmových zápletek na základě vybraného žánru a pozitivních aspektů z filmových recenzí.

## ✨ Vlastnosti

- 📄 **Nahrání a zpracování PDF**: Extrakce textu z PDF s filmovými recenzemi
- 🎭 **Výběr žánru**: 15 různých filmových žánrů na výběr
- 🔍 **Inteligentní vyhledávání**: Sémantické vyhledávání pozitivních aspektů z recenzí
- 📝 **Generování zápletek**: AI vytváří originální zápletky v češtině
- 🎬 **Inspirace recenzemi**: Zápletky inspirované tím, co lidé na filmech oceňují
- 📚 **Historie zápletek**: Uložení všech vygenerovaných zápletek

## 🚀 Instalace

1. **Nainstalujte závislosti:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Spusťte aplikaci:**
   ```bash
   streamlit run movie_plot_generator.py
   ```

## 📖 Jak používat

### 1. Nahrání dokumentu
- Klikněte na "Vyberte PDF soubor" v postranním panelu
- Nahrajte PDF s filmovými recenzemi
- Počkejte na zpracování dokumentu

### 2. Výběr žánru
Po zpracování dokumentu:
- Vyberte jeden z 15 dostupných žánrů pomocí radio buttonů:
  - Akční, Komedie, Drama, Horor, Sci-Fi, Fantasy
  - Romantický, Thriller, Krimi, Historický, Válečný
  - Životopisný, Dobrodružný, Western, Mysteriózní

### 3. Generování zápletky
- Klikněte na "🎬 Vygenerovat zápletku"
- Aplikace najde relevantní pozitivní aspekty z recenzí
- AI vygeneruje originální zápletku v češtině

### 4. Správa zápletek
- Všechny vygenerované zápletky se zobrazují s časovým razítkem
- Můžete vymazat všechny zápletky tlačítkem "🗑️ Vymazat všechny zápletky"

## 🎯 Jak to funguje

### Architektura
```
PDF recenze → Extrakce textu → Rozdělení na části → Embeddings → Vektorová databáze
                                                                        ↓
Výběr žánru → Sémantické vyhledávání → Kontext z recenzí → AI generování → Originální zápletka
```

### Komponenty

- **PDF zpracování**: PyMuPDF, pdfplumber a PyPDF2 pro robustní extrakci textu
- **Embeddings**: SentenceTransformer (all-MiniLM-L6-v2) pro sémantické reprezentace
- **Vektorová databáze**: FAISS pro rychlé vyhledávání podobnosti
- **AI integrace**: Granite-32-8B-Instruct model pro generování zápletek

### Konfigurace

Aplikace je nakonfigurována pro:
- **LLM Endpoint**: Granite-32-8B-Instruct
- **Embeddings**: all-MiniLM-L6-v2
- **Velikost částí**: 1000 znaků s překryvem 200 znaků
- **Teplota AI**: 0.8 pro kreativní generování
- **Maximální tokeny**: 1500 pro delší zápletky

## 📋 Systémový prompt

AI používá následující instrukce:
- Generuje POUZE v češtině
- Vytváří originální zápletky (ne kopie existujících filmů)
- Zápletky jsou dlouhé 200-300 slov
- Zahrnuje hlavní postavy, konflikt a náznač řešení
- Inspiruje se pozitivními aspekty z recenzí
- Nezmiňuje konkrétní názvy existujících filmů

## 🛠️ Řešení problémů

### Časté problémy

1. **Chyby importu**: Ujistěte se, že jsou nainstalovány všechny závislosti z requirements.txt
2. **PDF chyby**: Aplikace používá 3 různé knihovny pro extrakci PDF jako zálohu
3. **Připojení k API**: Zkontrolujte dostupnost LLM endpointu
4. **Paměťové problémy**: Pro velké dokumenty zvažte menší PDF soubory

### Tipy pro výkon

- **Menší PDF**: Fungují lépe pro rychlejší zpracování
- **Jasný text**: PDF s čitelným textem fungují nejlépe
- **Internetové připojení**: Potřebné pro stahování modelů a API volání

## 📚 Závislosti

- streamlit: Webové rozhraní
- PyPDF2, pdfplumber, PyMuPDF: Extrakce textu z PDF
- sentence-transformers: Textové embeddings
- faiss-cpu: Vektorové vyhledávání
- numpy: Numerické operace
- requests: HTTP API volání
- torch: Framework pro hluboké učení
- scikit-learn: Záložní embeddings

## 🎬 Ukázkové žánry

Aplikace podporuje tyto filmové žánry:
- **Akční**: Filmy plné akce a napětí
- **Komedie**: Vtipné a zábavné filmy
- **Drama**: Emocionálně bohaté příběhy
- **Horor**: Strašidelné a napínavé filmy
- **Sci-Fi**: Vědecko-fantastické příběhy
- **Fantasy**: Fantastické světy a magie
- **Romantický**: Příbjegy o lásce
- **Thriller**: Napínavé psychologické filmy
- **Krimi**: Detektivní a kriminální příběhy
- **Historický**: Filmy z různých historických období
- **Válečný**: Příběhy z válek a konfliktů
- **Životopisný**: Příběhy skutečných osob
- **Dobrodružný**: Filmy plné dobrodružství
- **Western**: Příběhy z Divokého západu
- **Mysteriózní**: Záhadné a tajemné příběhy

## 📄 Licence

Tento projekt je určen pro vzdělávací a demonstrační účely. 
