from flask import Flask, request, jsonify, render_template, send_file
from flask_caching import Cache
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import nltk
from word2number import w2n
import inflect
import random
from nltk.corpus import wordnet as wn
import PyPDF2
from werkzeug.utils import secure_filename
from html import escape
import torch
import logging
import re
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tempfile
from fpdf import FPDF
import io
import json
try:
    import wikipedia
except ImportError:
    wikipedia = None
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache'
cache = Cache(app)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    logger.info("Initializing question generation pipeline...")
    qg_pipeline = pipeline(
        "text2text-generation",
        model="valhalla/t5-base-e2e-qg",
        device=0 if torch.cuda.is_available() else -1
    )
    logger.info("Initializing question answering pipeline...")
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",  
        device=0 if torch.cuda.is_available() else -1
    )
except Exception as e:
    logger.error(f"Error initializing pipelines: {e}")
    logger.info("Falling back to smaller models...")
    qg_pipeline = pipeline("text2text-generation", model="t5-small")
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
try:
    logger.info("Initializing sentence transformer...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    logger.error(f"Error initializing sentence transformer: {e}")
def generate_pdf(questions):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50
    for i, q in enumerate(questions, 1):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{i}. {q['question']}")
        y -= 20
        if q['type'] == 'MCQ':
            c.setFont("Helvetica", 11)
            for opt in q['options']:
                c.drawString(70, y, f"- {opt}")
                y -= 15  
        y -= 20
        if y < 100:
            c.showPage()
            y = height - 50
    c.save()
    buffer.seek(0)
    return buffer
def clean_question(question):
    question = question.replace('<sep>', '').strip('?.').strip()
    if len(question) < 10:
        return None
    return question + '?' if not question.endswith('?') else question
def is_question_unique(new_question, existing_questions):
    if not existing_questions:
        return True
    try:
        new_embedding = sentence_model.encode(new_question)
        existing_embeddings = sentence_model.encode(list(existing_questions))
        similarities = util.cos_sim(new_embedding, existing_embeddings)[0]
        return all(sim < 0.8 for sim in similarities)
    except:
        return True
def is_scientist_related_question(question_text):
    scientist_related_keywords = [
        "who is the scientist","who","which","to whom" ,"with what", "who discovered", "who developed", "who invented",
        "who first used", "who pioneered", "who is known for", "who created",
        "what scientist", "which scientist", "which person", "who", "who invented",
        "who founded", "who was the first to", "who proposed", "which person"
    ]
    return any(re.search(r'\b' + re.escape(keyword) + r'\b', question_text.lower()) for keyword in scientist_related_keywords)
def generate_scientist_distractors(answer, num_distractors=3):
    related_scientists = [
        "Louis Pasteur", "Gregor Mendel", "Marie Curie", "Charles Darwin", "Barbara McClintock",
        "Richard Dawkins", "James Watson", "Francis Crick", "George Washington Carver", 
        "Robert Hooke", "Alexander Fleming", "Rosalind Franklin", "Craig Venter"
    ]
    distractors = []
    for name in related_scientists:
        if name != answer:  
            distractors.append(name)
        if len(distractors) >= num_distractors:
            break
    if len(distractors) < num_distractors:
        fallback_words = ["material", "device", "solution", "technique", "innovation", "discovery"]
        random.shuffle(fallback_words)
        for word in fallback_words:
            if word not in distractors:
                distractors.append(word)
            if len(distractors) >= num_distractors:
                break
    return distractors
def generate_distractors(answer, context, num_distractors=3):
    try:
        if not answer or not context:
            raise ValueError("Missing answer or context")
        answer_text = str(answer).strip().lower()
        distractors = set()
        if len(answer_text) < 1:
            raise ValueError("Answer text too short")
        if answer_text.replace('.', '', 1).isdigit():
            return handle_numeric_answer(answer_text, num_distractors)
            
        return handle_textual_answer(answer_text, context, num_distractors)
    except Exception as e:
        logger.error(f"Distractor generation error: {str(e)}")
        base = answer_text if isinstance(answer, str) else "option"
        return [f"{base.capitalize()} {i+1}" for i in range(num_distractors)]
p = inflect.engine()
def is_word_number(s):
    try:
        return p.number_to_words(p.words_to_number(s)) == s.lower()
    except:
        return False
def handle_numeric_answer(answer_text, num_distractors):
    distractors = set()
    try:
        if is_word_number(answer_text):
            value = p.words_to_number(answer_text)
            format_type = 'word'
        else:
            value = float(answer_text) if '.' in answer_text else int(answer_text)
            format_type = 'digit'
    except:
        return []
    magnitude = abs(value)
    if magnitude == 0:
        perturbations = [1, -1, 0.5, -0.5, 2, -2]
    elif magnitude < 1:
        scale = max(0.1, magnitude/10)
        perturbations = [scale, -scale, scale*2, -scale*2]
    elif magnitude < 10:
        perturbations = [1, -1, 2, -2, 0.5, -0.5]
    elif magnitude < 100:
        perturbations = [5, -5, 10, -10, 2, -2]
    else:
        scale = max(1, magnitude // 10)
        perturbations = [scale, -scale, scale*2, -scale*2]
    for p_val in perturbations:
        if len(distractors) >= num_distractors:
            break
        new_val = value + p_val
        if format_type == 'digit':
            distractor = str(int(new_val) if new_val == int(new_val) else round(new_val, 2))
        else:
            distractor = p.number_to_words(int(new_val))
        if distractor.lower() != answer_text.lower():
            distractors.add(distractor)
    return list(distractors)[:num_distractors]
def handle_textual_answer(answer_text, context, num_distractors):
    distractors = set()
    try:
        synonyms = set()
        antonyms = set()
        hypernyms = set()
        hyponyms = set()
        for syn in wn.synsets(answer_text):
            for lemma in syn.lemmas():
                clean_name = lemma.name().replace('_', ' ').lower()
                if clean_name != answer_text:
                    synonyms.add(clean_name)
                for antonym in lemma.antonyms():
                    clean_ant = antonym.name().replace('_', ' ').lower()
                    if clean_ant != answer_text:
                        antonyms.add(clean_ant)
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    clean_hyper = lemma.name().replace('_', ' ').lower()
                    if clean_hyper != answer_text:
                        hypernyms.add(clean_hyper)
            for hypo in syn.hyponyms():
                for lemma in hypo.lemmas():
                    clean_hypo = lemma.name().replace('_', ' ').lower()
                    if clean_hypo != answer_text:
                        hyponyms.add(clean_hypo)
        distractors.update(list(antonyms)[:1])
        if len(distractors) < num_distractors:
            distractors.update(list(hypernyms.union(hyponyms))[:num_distractors - len(distractors)])
        if len(distractors) < num_distractors:
            distractors.update(list(synonyms)[:num_distractors - len(distractors)])
    except Exception as e:
        logger.warning(f"WordNet processing failed: {str(e)}")
    if len(distractors) < num_distractors:
        try:
            sentences = nltk.sent_tokenize(context)
            words = nltk.word_tokenize(context.lower())
            pos_tags = nltk.pos_tag(words)
            candidate_terms = [
                word for word, pos in pos_tags 
                if pos in ['NN', 'NNS', 'NNP', 'NNPS'] 
                and word != answer_text
                and len(word) > 2
            ]
            term_freq = nltk.FreqDist(candidate_terms)
            top_terms = [term for term, _ in term_freq.most_common(10)]
            for term in top_terms:
                if len(distractors) >= num_distractors:
                    break
                if term not in distractors:
                    distractors.add(term)
        except Exception as e:
            logger.warning(f"Context processing failed: {str(e)}")
        if len(distractors) < num_distractors:
            general_fallbacks = [
                'scientist', 'engineer', 'researcher', 'device', 'machine', 'tool',
                'solution', 'method', 'concept', 'innovation', 'application',
                'experiment', 'reaction', 'model', 'component', 'system', 'strategy',
                'environment', 'organism', 'material', 'resource', 'mechanism'
            ]
            random.shuffle(general_fallbacks)
            for word in general_fallbacks:
                if len(distractors) >= num_distractors:
                    break
                if word.lower() != answer_text and word not in distractors:
                    distractors.add(word)
    final_distractors = []
    for d in list(distractors)[:num_distractors]:
        if d and isinstance(d, str):
            final_distractors.append(d.capitalize() if answer_text[0].isupper() else d)
    return final_distractors
@cache.memoize(timeout=3600)
def get_wikipedia_summary(topic, sentences):
    if not wikipedia:
        return topic
    try:
        return wikipedia.summary(topic, sentences=sentences, auto_suggest=False)
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=sentences)
    except Exception as e:
        logger.error(f"Error fetching Wikipedia summary: {e}")
        return topic
@app.route('/')
def serve_index():
    """Serve the index.html file."""
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file uploads for context."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        file = request.files['file']
        filename = secure_filename(file.filename)
        if filename.endswith('.pdf'):
            pdf = PyPDF2.PdfReader(file)
            context = ''.join(page.extract_text() or '' for page in pdf.pages)
        elif filename.endswith('.txt'):
            context = file.read().decode('utf-8')
        else:
            return jsonify({'error': 'Unsupported file format (use PDF or TXT)'}), 400
        return jsonify({'context': context[:10000]})  
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return jsonify({'error': 'Error processing file', 'details': str(e)}), 500
@app.route('/generate', methods=['POST'])
def generate_questions():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
        topic = escape(data.get('topic', '').strip()[:500])
        if not topic:
            return jsonify({'error': 'Topic cannot be empty or too long'}), 400
        qtype = data.get('qtype', 'MCQ')
        num_questions = min(max(int(data.get('num_questions', 5)), 1), 20)
        difficulty = data.get('difficulty', 'medium')
        context = data.get('context', topic)
        if context == topic and wikipedia:
            summary_sentences = {'easy': 5, 'medium': 10, 'hard': 20}.get(difficulty, 10)
            context = get_wikipedia_summary(topic, summary_sentences)
        response_questions = []
        max_attempts = num_questions * 3
        attempts = 0
        generated_questions = set()
        while len(response_questions) < num_questions and attempts < max_attempts:
            attempts += 1
            try:
                question_output = qg_pipeline(
                    context,
                    max_length=128,
                    do_sample=True,
                    top_k=100,
                    temperature=1.0,
                    num_return_sequences=1
                )
                question_text = question_output[0].get('generated_text', '').strip()
                question_list = [clean_question(q) for q in question_text.split('<sep>') if clean_question(q)]
                for question_text in question_list:
                    if question_text in generated_questions or not is_question_unique(question_text, generated_questions):
                        continue
                    generated_questions.add(question_text)
                    answer_result = qa_pipeline({
                        'question': question_text,
                        'context': context
                    }, max_answer_len=50)
                    answer = answer_result.get('answer', '').strip()
                    if not answer or answer_result.get('score', 0.0) < 0.5:
                        continue
                    if qtype == 'MCQ':
                        if is_scientist_related_question(question_text):  # Name-related question
                            distractors = generate_scientist_distractors(answer, num_distractors=3)
                        elif question_text.lower().startswith("how many"):
                            distractors = handle_numeric_answer(answer, num_distractors=3)
                        else:
                            distractors = generate_distractors(answer, context, num_distractors=3)
                        options = distractors + [answer]
                        random.shuffle(options)
                        options = list(dict.fromkeys([opt.strip() for opt in options if opt.strip() and not opt.startswith('Option')]))
                        while len(options) < 4:
                            fallback_words = [
                                "scientists", "biotechnologists", "researchers", "engineers", "biologists", "geneticists", "innovators", "technicians", 
                                "doctors", "chemists", "medical experts", "lab assistants", "physicists", "mathematicians", "statisticians", "programmers", 
                                "developers", "analysts", "consultants", "technologists", "physiologists", "pathologists", "surgeons", "nurses", "neurologists", 
                                "oncologists", "hematologists", "genetic counselors", "epidemiologists", "biochemists", "pharmacologists", "immunologists", "toxicologists",
                                "biotechnology", "genetic engineering", "bioinformatics", "molecular biology", "bioengineering", "bioreactors", "enzymes", 
                                "genetically modified organisms", "biosensors", "DNA sequencing", "gene editing", "cloning", "microorganisms", "biofuels", "bioremediation",
                                "pollution control", "carbon capture", "waste treatment", "biodegradation", "biomass", "sustainability", "green technology", 
                                "ecosystem restoration", "recycling", "renewable energy", "environmental cleanup", "climate change", "environmental science",
                                "ecology", "plant science", "animal science", "aquaculture", "genomics", "epigenetics", "synthetic biology", "chemoinformatics",
                                "pharmacogenomics", "medicinal chemistry", "biological engineering", "environmental engineering", "nanotechnology", "material science",
                                "robotics", "computational biology", "bioengineering", "systems biology", "chemistry", "physics", "astronomy", "geology",
                                "innovation", "advancement", "technology", "solution", "progress", "discovery", "breakthrough", "methodology", "approach", 
                                "strategy", "system", "framework", "model", "paradigm", "principle", "mechanism", "reaction", "process", "theory", "hypothesis", 
                                "analysis", "statistical analysis", "simulation", "prediction", "calculation", "observation", "experiment", "data collection", 
                                "visualization", "diagnosis", "screening", "treatment", "prevention", "analysis", "interpretation", "evaluation", "validation", 
                                "optimization", "regulation", "testing", "monitoring", "assessment", "results", "impact", "applications", "uses", "products", "outputs", "findings",
                                "bacteria", "viruses", "fungi", "algae", "cells", "organisms", "microbes", "genomes", "species", "plants", "animals", "yeasts", 
                                "insects", "worms", "mammals", "reptiles", "birds", "amphibians", "marine life", "flora", "fauna", "microflora", "plankton", 
                                "zooplankton", "phytoplankton", "protozoa", "fungus", "bacteria species", "viruses", "pathogens", "symbionts", "ecosystem", "biome", "biota",
                                "materials", "tools", "machines", "chemicals", "resources", "devices", "systems", "equipment", "instruments", "gadgets", "software", 
                                "hardware", "technologies", "platforms", "components", "elements", "structures", "medicines", "therapies", "vaccines", "implants", "prototypes", 
                                "molecules", "substances", "compounds", "formulations", "substrates", "agents", "nanomaterials", "artificial intelligence", "machine learning", 
                                "deep learning", "algorithms", "chips", "circuits", "networks", "batteries", "energy sources", "fuel", "data", "computers", "computing", "algorithms",
                                "sustainability", "green technology", "ecosystem restoration", "recycling", "renewable energy", "pollution control", "climate change", 
                                "global warming", "environmental health", "conservation", "environmental protection", "eco-friendly", "clean energy", "natural resources", 
                                "biomass", "carbon emissions", "carbon footprint", "reforestation", "pollutants", "waste management", "greenhouse gases", "ecological balance", 
                                "oceanography", "soil health", "air quality", "water quality", "biodiversity", "natural habitat", "conservation biology", "habitat loss", "deforestation", 
                                "carbon sequestration", "climate action", "environmental policy", "green innovation", "ecotourism",
                                "products", "solutions", "applications", "uses", "benefits", "effects", "results", "outcomes", "goals", "targets", "findings", "conclusions", 
                                "inventions", "revolutions", "transformations", "discoveries", "innovations", "scientific progress", "social impact", "outcomes", "findings", 
                                "insights", "achievements", "research", "development", "engineering solutions", "technological advances", "business applications", "scientific solutions", 
                                "strategies", "challenges", "opportunities", "perspectives", "studies", "reports", "papers", "publications"
                            ]
                            random.shuffle(fallback_words)
                            for word in fallback_words:
                                candidate = word.capitalize() if answer[0].isupper() else word
                                if candidate not in options and candidate.lower() != answer.lower():
                                    options.append(candidate)
                                if len(options) >= 4:
                                    break
                        options = options[:4] 
                        question_entry = {
                            'type': 'MCQ',
                            'question': question_text,
                            'options': options,
                            'answer': answer
                        }
                    elif qtype == 'Short Answer':
                        question_entry = {
                            'type': 'Short Answer',
                            'question': question_text,
                            'answer': answer
                        }  
                    response_questions.append(question_entry)    
            except Exception as e:
                logger.error(f"Error generating question (attempt {attempts}): {e}")
                continue
        if not response_questions:
            return jsonify({'error': 'Could not generate any valid questions'}), 400 
        return jsonify({
            'questions': response_questions,
            'context': context[:500] + "..." if len(context) > 500 else context
        })  
    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({'error': 'An error occurred while generating questions', 'details': str(e)}), 500
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    topic = request.form.get('topic', 'Quiz')
    questions = json.loads(request.form.get('questions', '[]'))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Topic: {topic}", align="L")
    pdf.ln()
    for i, q in enumerate(questions, 1):
        pdf.multi_cell(0, 10, f"{i}. {q['question']}", align="L")
        if q['type'] == 'MCQ':
            for opt in q.get('options', []):
                prefix = "- "
                pdf.multi_cell(0, 10, f"{prefix}{opt}", align="L")
        pdf.ln()
    if any(q['type'] == 'MCQ' for q in questions):
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Answer Key", ln=True, align="L")
        pdf.set_font("Arial", size=12)
        for i, q in enumerate(questions, 1):
            if q['type'] == 'MCQ':
                answer = q.get('answer', 'Not provided')
                pdf.multi_cell(0, 10, f"{i}. {answer}", align="L")
    pdf_path = "quiz.pdf"
    pdf.output(pdf_path)
    return send_file(pdf_path, as_attachment=True)
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)