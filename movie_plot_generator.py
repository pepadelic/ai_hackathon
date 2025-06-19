import streamlit as st
import requests
import PyPDF2
import io
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict
import json
import time
from datetime import datetime

# Configuration
URL = "https://granite-32-8b-instruct-apodhrad-test.apps.cluster-7r2vd.7r2vd.sandbox1120.opentlc.com"
CHAT_URL = f"{URL}/v1/chat/completions"

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'generated_plots' not in st.session_state:
    st.session_state.generated_plots = []

@st.cache_resource
def load_embeddings_model():
    """Load the sentence transformer model for embeddings with error handling"""
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*Torch.*")
        warnings.filterwarnings("ignore", message=".*torch.*")
        
        import torch
        # Set torch to use minimal warnings
        torch.set_warn_always(False)
        
        st.info("🔄 Načítám model pro vektorová embeddings (all-MiniLM-L6-v2)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("✅ Model pro embeddings byl úspěšně načten!")
        return model
    except Exception as e:
        st.error(f"❌ Nepodařilo se načíst SentenceTransformer model: {str(e)}")
        st.info("💡 Zkouším alternativní přístup s TF-IDF...")
        
        try:
            # Fallback to a simpler approach using sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            st.success("✅ Používám TF-IDF vektorizátor jako náhradu")
            return TfidfVectorizer(max_features=1000, stop_words='english')
        except ImportError:
            st.error("❌ Nepodařilo se načíst žádný model pro embeddings. Nainstalujte potřebné závislosti.")
            return None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text content from uploaded PDF file with multiple fallback methods"""
    text = ""
    successful_pages = 0
    total_pages = 0
    
    # Reset file pointer to beginning
    pdf_file.seek(0)
    
    # Method 1: Try PyMuPDF first (most robust)
    try:
        import fitz  # PyMuPDF
        pdf_file.seek(0)
        
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = pdf_document.page_count
        
        st.info(f"📄 Zpracovávám PDF s {total_pages} stránkami pomocí PyMuPDF...")
        
        for page_num in range(total_pages):
            try:
                page = pdf_document[page_num]
                page_text = page.get_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    successful_pages += 1
            except Exception as page_error:
                st.warning(f"⚠️ Nepodařilo se extrahovat text ze stránky {page_num + 1} s PyMuPDF: {str(page_error)}")
                continue
        
        pdf_document.close()
        
        if text.strip():
            st.success(f"✅ Text extrahován pomocí PyMuPDF ({successful_pages}/{total_pages} stránek úspěšně)")
            return text
            
    except ImportError:
        st.info("💡 Instaluji PyMuPDF pro lepší podporu PDF...")
    except Exception as e:
        st.warning(f"⚠️ PyMuPDF extrakce selhala: {str(e)}. Zkouším alternativní metodu...")
    
    # Method 2: Try with pdfplumber as fallback
    try:
        import pdfplumber
        pdf_file.seek(0)
        text = ""  # Reset text
        successful_pages = 0
        
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            st.info(f"📄 Zkouším pdfplumber extrakci pro {total_pages} stránek...")
            
            for page_num, page in enumerate(pdf.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                        successful_pages += 1
                except Exception as page_error:
                    st.warning(f"⚠️ Nepodařilo se extrahovat text ze stránky {page_num + 1} s pdfplumber: {str(page_error)}")
                    continue
        
        if text.strip():
            st.success(f"✅ Text extrahován pomocí pdfplumber ({successful_pages}/{total_pages} stránek úspěšně)")
            return text
            
    except ImportError:
        st.info("💡 pdfplumber není dostupný, zkouším PyPDF2...")
    except Exception as e:
        st.warning(f"⚠️ pdfplumber extrakce také selhala: {str(e)}. Zkouším PyPDF2...")
    
    # Method 3: Try PyPDF2 as last resort with better error handling
    try:
        pdf_file.seek(0)
        text = ""  # Reset text
        successful_pages = 0
        
        # Try different PyPDF2 approaches
        pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)  # Use non-strict mode
        
        # Check if PDF is encrypted
        if pdf_reader.is_encrypted:
            st.error("❌ PDF je chráněno heslem. Nahrajte nechráněný PDF.")
            return ""
        
        total_pages = len(pdf_reader.pages)
        st.info(f"📄 Zkouším PyPDF2 extrakci pro {total_pages} stránek (non-strict mód)...")
        
        # Extract text from all pages with better error handling
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                # Try multiple extraction methods for each page
                page_text = ""
                
                # Method 3a: Standard extraction
                try:
                    page_text = page.extract_text()
                except:
                    pass
                
                # Method 3b: Try extracting with different parameters
                if not page_text or not page_text.strip():
                    try:
                        page_text = page.extract_text(extraction_mode="layout")
                    except:
                        pass
                
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    successful_pages += 1
                else:
                    st.warning(f"⚠️ Stránka {page_num + 1} se zdá být prázdná nebo obsahuje pouze obrázky")
                    
            except Exception as page_error:
                # Don't show warnings for every page failure with PyPDF2, just count them
                continue
        
        if text.strip():
            st.success(f"✅ Text extrahován pomocí PyPDF2 ({successful_pages}/{total_pages} stránek úspěšně)")
            return text
        elif successful_pages == 0:
            st.warning("⚠️ PyPDF2 nemohl extrahovat text z žádné stránky")
            
    except Exception as e:
        st.warning(f"⚠️ PyPDF2 extrakce také selhala: {str(e)}")
    
    # If all methods failed but we got some text
    if text.strip():
        st.success(f"✅ Částečná extrakce textu úspěšná ({successful_pages}/{total_pages} stránek)")
        return text
    
    # If completely failed
    st.error("""
    ❌ **Nepodařilo se extrahovat text z PDF**
    
    **Možné důvody:**
    - PDF obsahuje pouze obrázky/skenovaný obsah (žádný extrahovatelný text)
    - PDF má neobvyklé kódování nebo je vážně poškozený
    - PDF používá nestandardní fonty nebo složité formátování
    
    **Návrhy:**
    - Zkuste jiný PDF soubor se standardním textovým obsahem
    - Použijte OCR nástroje pro převod skenovaných PDF na text-prohledávatelné PDF
    - Zkuste otevřít PDF v jiné aplikaci a znovu jej uložit
    - Zkontrolujte, zda se PDF správně zobrazuje v prohlížeči PDF
    """)
    return ""

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            break_point = max(last_period, last_newline)
            
            if break_point > start + chunk_size // 2:
                chunk = text[start:break_point + 1]
                end = break_point + 1
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def create_vector_store(chunks: List[str], model) -> faiss.IndexFlatIP:
    """Create FAISS vector store from text chunks"""
    if not chunks:
        return None, []
    
    # Generate embeddings
    with st.spinner("Vytvářím embeddings..."):
        embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, embeddings

def retrieve_relevant_chunks(query: str, model, vector_store, chunks: List[str], k: int = 5) -> List[str]:
    """Retrieve most relevant chunks for a given query"""
    if not vector_store or not chunks:
        return []
    
    # Generate query embedding
    query_embedding = model.encode([query])
    query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
    
    # Search for similar chunks
    scores, indices = vector_store.search(query_embedding.astype('float32'), k)
    
    # Return relevant chunks
    relevant_chunks = []
    for i, idx in enumerate(indices[0]):
        if scores[0][i] > 0.1:  # Similarity threshold
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

def call_llm(messages: List[Dict], max_retries: int = 5) -> str:
    """Call the LLM API with enhanced retry logic and exponential backoff"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Try with different configurations for better success rate
    configurations = [
        {
            "model": "granite-32-8b-instruct",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1500,
            "stream": False
        },
        {
            "model": "granite-32-8b-instruct", 
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        },
        {
            "model": "granite-32-8b-instruct",
            "messages": messages, 
            "temperature": 0.6,
            "max_tokens": 800,
            "stream": False
        }
    ]
    
    for config_idx, payload in enumerate(configurations):
        st.info(f"🔄 Zkouším konfiguraci {config_idx + 1}/3...")
        
        for attempt in range(max_retries):
            try:
                # Exponential backoff delay
                if attempt > 0:
                    delay = min(2 ** attempt, 30)  # Max 30 seconds
                    st.info(f"⏳ Čekám {delay} sekund před dalším pokusem...")
                    time.sleep(delay)
                
                # Adjust timeout based on attempt
                timeout = 30 + (attempt * 15)  # Start with 30s, increase each attempt
                
                st.info(f"📡 Odesílám požadavek na LLM API (pokus {attempt + 1}/{max_retries}, timeout: {timeout}s)...")
                
                response = requests.post(
                    CHAT_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Úspěšně získána odpověď od LLM!")
                    return result['choices'][0]['message']['content']
                elif response.status_code == 504:
                    st.warning(f"⚠️ Server timeout (504) - pokus {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        st.error("❌ Server je přetížený nebo nedostupný. Zkuste to za chvíli.")
                    continue
                elif response.status_code == 503:
                    st.warning(f"⚠️ Služba nedostupná (503) - pokus {attempt + 1}/{max_retries}")
                    continue
                elif response.status_code == 502:
                    st.warning(f"⚠️ Bad Gateway (502) - pokus {attempt + 1}/{max_retries}")
                    continue
                else:
                    st.warning(f"⚠️ API chyba: {response.status_code} - pokus {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        return f"""❌ **Nepodařilo se získat odpověď od LLM**
                        
**Stav chyby:** {response.status_code}
**Popis:** {response.text if len(response.text) < 200 else response.text[:200] + '...'}

**Co můžete zkusit:**
- Zkuste to za několik minut znovu
- Vyberte jiný žánr
- Zkontrolujte připojení k internetu
- Server může být přetížený"""
                    
            except requests.exceptions.Timeout:
                st.warning(f"⏰ Časový limit požadavku překročen (pokus {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1 and config_idx == len(configurations) - 1:
                    return """❌ **Časový limit požadavku překročen**
                    
**Možné řešení:**
- Server je příliš pomalý nebo přetížený
- Zkuste to za několik minut
- Možná je problém s připojením k internetu
- Kontaktujte správce API pokud problém přetrvává"""
                    
            except requests.exceptions.ConnectionError:
                st.warning(f"🔌 Chyba připojení (pokus {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1 and config_idx == len(configurations) - 1:
                    return """❌ **Chyba připojení k serveru**
                    
**Možné řešení:**
- Zkontrolujte připojení k internetu
- Server může být dočasně nedostupný
- Zkuste to za několik minut
- Možná je problém s API endpointem"""
                    
            except Exception as e:
                st.warning(f"❓ Neočekávaná chyba: {str(e)} (pokus {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1 and config_idx == len(configurations) - 1:
                    return f"""❌ **Neočekávaná chyba**
                    
**Popis chyby:** {str(e)}

**Co můžete zkusit:**
- Zkuste to znovu za chvíli
- Restartujte aplikaci
- Zkontrolujte, zda jsou nainstalovány všechny závislosti"""
    
    return """❌ **Nepodařilo se vygenerovat zápletku**
    
Všechny pokusy o spojení s AI selhal. Zkuste to prosím později nebo kontaktujte správce systému."""

def generate_fallback_plot(genre: str) -> str:
    """Generate a simple fallback plot when API is unavailable"""
    import random
    
    # Simple plot templates for each genre
    plot_templates = {
        "Akční": [
            "Bývalý voják {name1} musí zachránit svou unesenou dceru z rukou nebezpečného zločineckého syndikátu. S pomocí své kolegyně {name2} se vydává na nebezpečnou misi plnou střelby a honů. Postupně odhaluje, že za únosem stojí jeho bývalý přítel, který se stal zrádcem.",
            "Policejní detektiv {name1} objeví spiknutí proti vládě. Společně s reportérkou {name2} se snaží odhalit pravdu, zatímco je pronásledují nájemní vrazi. Napínavá honička končí velkým odhalením korupce na nejvyšších místech."
        ],
        "Komedie": [
            "Neúspěšný herec {name1} se omylem vydává za slavného režiséra. S pomocí své přítelkyně {name2} se snaží udržet tuto lež, což vede k řadě vtipných situací. Nakonec díky své upřímnosti získá skutečnou filmovou roli.",
            "Dva sousedé {name1} a {name2} se neustále hádají kvůli maličkostem. Když musí společně organizovat sousedskou slavnost, jejich spory vedou k hilariózním nedorozuměním, ale nakonec se z nich stanou nejlepší přátelé."
        ],
        "Drama": [
            "Mladá učitelka {name1} přichází do problémové školy v předměstí. Postupně získává důvěru svých studentů, včetně rebelského {name2}, a pomáhá jim najít cestu k lepší budoucnosti. Její odhodlání mění životy celé komunity.",
            "Otec {name1} se po letech snaží znovu navázat vztah se svým odcizeným synem {name2}. Jejich společná cesta je plná emocí, vzpomínek a postupného odpuštění. Nakonec pochopí, jak moc si navzájem chybí."
        ],
        "Horor": [
            "Skupina přátel se nastěhuje do starého domu, který skrývá temné tajemství. Postupně descobují, že dům je prokletý a jeho předchozí obyvatelé zde zahynuli za záhadných okolností. {name1} musí najít způsob, jak prokletí zlomit.",
            "V malém městečku se začnou dít podivné věci. Psycholog {name1} vyšetřuje sérii záhadných zmizení a zjišťuje, že za vším stojí nadpřirozená síla. S pomocí místní knihovnice {name2} hledá způsob, jak zlo zastavit."
        ],
        "Sci-Fi": [
            "V roce 2150 objeví vědkyně {name1} portál do paralelního vesmíru. Společně s kolegou {name2} prozkoumává alternativní realitu, kde lidstvo vyvinulo odlišnou technologii. Musí se rozhodnout, zda zůstat nebo se vrátit domů.",
            "Astronaut {name1} se po dlouhé misi vrací na Zemi a zjišťuje, že se planeta změnila k nepoznání. S pomocou rebelky {name2} se snaží pochopit, co se stalo, a bojuje za obnovu původního světa."
        ],
        "Fantasy": [
            "Mladý čaroděj {name1} objeví, že je potomkem mocné čarodějné dynastie. S průvodkyní {name2} se vydává na nebezpečnou cestu, aby získal své dědictví a porazil temného mága, který ohrožuje království.",
            "V světě, kde magie mizí, poslední čarodějnice {name1} musí najít způsob, jak ji zachránit. Společně s elfským válečníkem {name2} hledá pradávný artefakt, který může obnovit magickou rovnováhu."
        ],
        "Romantický": [
            "Úspěšná architektka {name1} se vrací do rodného městečka a znovu potkává svou středoškolskou lásku {name2}. Jejich cesty se rozdělily, ale osudy je znovu spojují. Musí se rozhodnout mezi kariérou a láskou.",
            "Dva lidé {name1} a {name2} se potkávají každý den v kavárně, aniž by spolu mluvili. Když se konečně odváží promluvit, zjistí, že mají mnoho společného a začíná krásný příběh lásky."
        ],
        "Thriller": [
            "Psycholožka {name1} léčí pacienta s poruchou paměti, ale postupně zjišťuje, že je svědkem vraždy. S detektivem {name2} se snaží odhalit pravdu, zatímco ji někdo pronásleduje a ohrožuje.",
            "Novinář {name1} vyšetřuje sérii záhadných úmrtí. Stopy vedou k mocné korporaci, kde pracuje jeho přítelkyně {name2}. Musí zjistit pravdu, aniž by ohrozil sebe i ji."
        ],
        "Krimi": [
            "Detektiv {name1} vyšetřuje vraždu, která připomíná případ z jeho minulosti. S novou partnerkou {name2} objevuje souvislosti, které vedou k osobní vendettě. Musí čelit vlastním démonům.",
            "Bývalý zločinec {name1} se snaží začít nový život, ale jeho minulost ho dostihne. S pomocí advokátky {name2} bojuje proti falešným obviněním a hledá skutečného viníka."
        ],
        "Historický": [
            "V době českého národního obrození se mladý spisovatel {name1} zapojuje do boje za nezávislost. S vlastenkou {name2} organizují tajné setkání a šíří zakázané texty. Jejich láska překonává všechny překážky.",
            "Během druhé světové války pomáhá lékařka {name1} partyzánům. S důstojníkem {name2} zachraňuje raněné a ukrývá pronásledované. Jejich odvaha mění osudy mnoha lidí."
        ],
        "Válečný": [
            "Mladý voják {name1} se během války ocitá odříznut od své jednotky. S místní obyvatelkou {name2} se snaží přežít v nepřátelském území. Jejich společný boj odhaluje skutečnou povahu humanity.",
            "Veterán {name1} se vrací z války změněný. S pomocí terapeutky {name2} se snaží vyrovnat s traumatem a najít místo v civilním světě. Jeho cesta k uzdravení je dlouhá ale povzbudivá."
        ],
        "Životopisný": [
            "Příběh výjimečné ženy {name1}, která přes všechny překážky dosáhla svého snu. S podporou svého přítele {name2} překonává společenské předsudky a stává se průkopnicí ve svém oboru.",
            "Mladý muž {name1} z chudých poměrů se díky svému talentu a houževnatosti dostává na vrchol. Jeho přítelkyně {name2} ho podporuje ve všech těžkých chvílích. Je to příběh o síle vůle a lásce."
        ],
        "Dobrodružný": [
            "Archeolog {name1} objeví mapu vedoucí k ztracenému pokladu. Společně s odvážnou průvodkyní {name2} se vydává na nebezpečnou expedici. Musí čelit přírodním katastrofám i zlým konkurentům.",
            "Pilot {name1} nouzově přistává na nezmapovaném ostrově. S biologkou {name2} objevuje neznámé druhy a snaží se najít cestu domů. Jejich dobrodružství odhaluje krásy i nebezpečí divočiny."
        ],
        "Western": [
            "Osamělý pistolník {name1} přijíždí do městečka ovládaného bandity. S pomocí odvážné majitelky salonu {name2} organizuje odpor proti tyranii. Jejich boj za spravedlnost mění celé město.",
            "Šerif {name1} se musí postavit gangu, který terorizuje okolí. Jeho partnerka {name2} mu pomáhá udržet pořádek. Je to příběh o odvaze a věrnosti zákonu."
        ],
        "Mysteriózní": [
            "Detektiv {name1} vyšetřuje sérii podivných událostí v malém městě. S knihovnicí {name2} odhaluje tajemství, které sahá hluboko do historie. Pravda je překvapivější, než čekali.",
            "Psycholog {name1} studuje paranormální jevy a setkává se s médiem {name2}. Společně zkoumají nevysvětlitelné úkazy a postupně odhalují hranice mezi realitou a nadpřirozenem."
        ]
    }
    
    # Generate random names
    czech_names = ["Anna", "Petr", "Marie", "Jan", "Eva", "Tomáš", "Věra", "Pavel", "Jana", "Martin", "Lucie", "Jiří", "Tereza", "David", "Kateřina"]
    name1 = random.choice(czech_names)
    name2 = random.choice([n for n in czech_names if n != name1])
    
    # Select template and format
    if genre in plot_templates:
        template = random.choice(plot_templates[genre])
        return template.format(name1=name1, name2=name2)
    else:
        return f"Příběh o {name1} a {name2} v žánru {genre} - bohužel nemám k dispozici šablonu pro tento žánr."

def generate_movie_plot(genre: str, good_reviews_context: str) -> dict:
    """Generate a movie plot based on genre and good reviews context"""
    
    # System prompt in Czech
    system_prompt = """Jsi kreativní scenárista, který vytváří zajímavé a originální filmové zápletky v češtině. 
    Tvým úkolem je vytvořit poutavý příběh filmu na základě zvoleného žánru a inspirovat se pozitivními recenzemi z poskytnutého kontextu.
    
    Pravidla:
    - Vždy piš POUZE v češtině
    - Vytvoř originální zápletku (ne kopii existujícího filmu)
    - Zápletka by měla být dlouhá asi 200-300 slov
    - Zahrň hlavní postavy, základní konflikt a náznač řešení
    - Inspiruj se pozitivními aspekty z recenzí (co lidé na filmech oceňují)
    - Buď kreativní a zajímavý
    - Nezmiňuj konkrétní názvy existujících filmů
    """
    
    user_prompt = f"""Žánr filmu: {genre}

Kontext z pozitivních recenzí:
{good_reviews_context}

Vytvoř originální zápletku pro nový film v žánru {genre}. Inspiruj se pozitivními aspekty z recenzí - co lidé na filmech oceňují (např. dobré vztahy mezi postavami, napětí, humor, emoce, překvapivé zvraty, atd.).

Zápletka filmu:"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Prepare full prompt for debugging
    full_prompt = f"**SYSTEM PROMPT:**\n{system_prompt}\n\n**USER PROMPT:**\n{user_prompt}"
    
    # Try to get AI-generated plot
    ai_response = call_llm(messages)
    
    # If AI response contains error messages, use fallback
    if ai_response.startswith("❌") or "Chyba:" in ai_response or "Nepodařilo se" in ai_response:
        st.warning("⚠️ AI služba není dostupná, používám záložní generátor zápletek...")
        fallback_plot = generate_fallback_plot(genre)
        final_plot = f"""🤖 **Záložní zápletka (AI nedostupná)**

{fallback_plot}

---
*Poznámka: Tato zápletka byla vygenerována záložním systémem, protože AI služba není momentálně dostupná. Pro kvalitnější výsledky zkuste později, až bude AI služba opět funkční.*"""
        
        return {
            'plot': final_plot,
            'raw_prompt': full_prompt,
            'raw_response': ai_response,
            'is_fallback': True
        }
    
    return {
        'plot': ai_response,
        'raw_prompt': full_prompt,
        'raw_response': ai_response,
        'is_fallback': False
    }

def main():
    st.set_page_config(
        page_title="Generátor filmových zápletek",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Generátor filmových zápletek")
    st.markdown("Nahrajte PDF s filmovými recenzemi a vygenerujte originální zápletku nového filmu na základě zvoleného žánru!")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Nahrání dokumentu")
        
        uploaded_file = st.file_uploader(
            "Vyberte PDF soubor",
            type="pdf",
            help="Nahrajte PDF s filmovými recenzemi"
        )
        
        if uploaded_file is not None:
            if not st.session_state.document_uploaded or st.button("Zpracovat dokument"):
                with st.spinner("Zpracovávám dokument..."):
                    # Load embeddings model
                    if st.session_state.embeddings_model is None:
                        st.session_state.embeddings_model = load_embeddings_model()
                    
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Chunk the text
                        chunks = chunk_text(text)
                        st.session_state.document_chunks = chunks
                        
                        # Create vector store
                        vector_store, embeddings = create_vector_store(
                            chunks, st.session_state.embeddings_model
                        )
                        st.session_state.vector_store = vector_store
                        
                        st.session_state.document_uploaded = True
                        st.success(f"✅ Dokument zpracován! Vytvořeno {len(chunks)} částí.")
                        
                        # Show document stats
                        st.info(f"""
                        **Statistiky dokumentu:**
                        - Celkem znaků: {len(text):,}
                        - Počet částí: {len(chunks)}
                        - Průměrná velikost části: {len(text)//len(chunks) if chunks else 0} znaků
                        """)
                    else:
                        st.error("Nepodařilo se extrahovat text z PDF")
        
        # Clear document button
        if st.session_state.document_uploaded:
            if st.button("🗑️ Vymazat dokument"):
                st.session_state.vector_store = None
                st.session_state.document_chunks = []
                st.session_state.document_uploaded = False
                st.session_state.generated_plots = []
                st.rerun()
    
    # Main plot generation interface
    if st.session_state.document_uploaded:
        st.success("📖 Dokument načten a připraven pro generování zápletek!")
        
        # Genre selection
        st.subheader("🎭 Výběr žánru filmu")
        
        genres = [
            "Akční", "Komedie", "Drama", "Horor", "Sci-Fi", "Fantasy", 
            "Romantický", "Thriller", "Krimi", "Historický", "Válečný",
            "Životopisný", "Dobrodružný", "Western", "Mysteriózní"
        ]
        
        selected_genre = st.radio(
            "Vyberte žánr pro nový film:",
            genres,
            horizontal=True
        )
        
        # Generate plot button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🎬 Vygenerovat zápletku", type="primary"):
                with st.spinner(f"Vytvářím originální zápletku pro {selected_genre.lower()} film..."):
                    # Get relevant context from reviews
                    genre_query = f"pozitivní recenze {selected_genre.lower()} film dobrý kvalitní oceňovaný"
                    relevant_chunks = retrieve_relevant_chunks(
                        genre_query,
                        st.session_state.embeddings_model,
                        st.session_state.vector_store,
                        st.session_state.document_chunks,
                        k=5
                    )
                    
                    context = "\n\n".join(relevant_chunks) if relevant_chunks else "Žádný relevantní kontext nebyl nalezen."
                    
                    # Generate plot
                    plot_data = generate_movie_plot(selected_genre, context)
                    
                    # Add to generated plots
                    timestamp = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
                    st.session_state.generated_plots.append({
                        'genre': selected_genre,
                        'plot': plot_data['plot'],
                        'raw_prompt': plot_data['raw_prompt'],
                        'raw_response': plot_data['raw_response'],
                        'is_fallback': plot_data['is_fallback'],
                        'timestamp': timestamp,
                        'context_used': context
                    })
                    
                    st.rerun()
        
        with col2:
            if st.button("🗑️ Vymazat všechny zápletky"):
                st.session_state.generated_plots = []
                st.rerun()
        
        # Display generated plots
        if st.session_state.generated_plots:
            st.subheader("📝 Vygenerované zápletky")
            
            for i, plot_data in enumerate(reversed(st.session_state.generated_plots)):
                with st.container():
                    st.markdown(f"### 🎬 {plot_data['genre']} film ({plot_data['timestamp']})")
                    st.markdown(plot_data['plot'])
                    
                    # Add expanders for debugging information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("🔍 Zobrazit raw prompt"):
                            st.markdown("**Prompt odeslaný do LLM:**")
                            st.text(plot_data.get('raw_prompt', 'Prompt není k dispozici'))
                            
                            if 'context_used' in plot_data:
                                st.markdown("**Kontext z recenzí:**")
                                st.text(plot_data['context_used'])
                    
                    with col2:
                        with st.expander("🤖 Zobrazit raw odpověď LLM"):
                            st.markdown("**Raw odpověď od LLM:**")
                            st.text(plot_data.get('raw_response', 'Odpověď není k dispozici'))
                            
                            # Show additional info
                            if plot_data.get('is_fallback', False):
                                st.warning("⚠️ Tato zápletka byla vygenerována záložním systémem")
                            else:
                                st.success("✅ Zápletka vygenerována pomocí AI")
                    
                    st.markdown("---")
    
    else:
        st.info("👆 Nahrajte prosím PDF dokument s filmovými recenzemi v postranním panelu!")
        
        # Show example usage
        st.markdown("""
        ### Jak používat tuto aplikaci:
        1. **Nahrajte PDF** s filmovými recenzemi pomocí nahrávače souborů v postranním panelu
        2. **Počkejte na zpracování** - aplikace extrahuje text a vytvoří embeddings
        3. **Vyberte žánr** filmu z nabídky
        4. **Klikněte na tlačítko** pro vygenerování zápletky
        5. **Získejte originální zápletku** inspirovanou pozitivními aspekty z recenzí
        
        ### Vlastnosti:
        - 📄 Extrakce textu z PDF a rozdělení na části
        - 🔍 Sémantické vyhledávání pomocí embeddings
        - 🎭 Výběr z 15 filmových žánrů
        - 📝 Generování originálních zápletek v češtině
        - 🎬 Inspirace pozitivními aspekty z recenzí
        """)

if __name__ == "__main__":
    main() 