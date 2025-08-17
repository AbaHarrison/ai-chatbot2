from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import secrets
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import openai
import os
from dotenv import load_dotenv
import pdfminer.high_level
from docx import Document
from job_fetcher import find_jobs_from_sentence, preload_job_embeddings, get_all_jobs
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import logging
import json
from typing import Optional, Dict, List
import re

security = HTTPBearer()
API_TOKEN = os.getenv("API_TOKEN")
if not API_TOKEN:
    raise Exception("API_TOKEN not found in environment variables")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_TOKEN:
        logger.warning("Invalid token attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app first
app = FastAPI(title="Motherboard Career Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-squarespace-domain.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {e}")
    raise

@app.on_event("startup")
async def startup_event():
    """Initialize data and connections on startup"""
    try:
        logger.info("Starting job data initialization...")
        preload_job_embeddings()  # This will populate the jobs_data global variable
        jobs = get_all_jobs()  # Now this will work since jobs_data is populated
        logger.info(f"Successfully loaded {len(jobs)} jobs")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        logger.error(f"Error details: {str(e)}")
        raise

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
try:
    creds = ServiceAccountCredentials.from_json_keyfile_name("gcreds.json", scope)
    gs_client = gspread.authorize(creds)
except Exception as e:
    logger.error(f"Failed to initialize Google Sheets client: {e}")



# FAQ knowledge base
FAQ_DATA = {
    "cv_gaps": "Career gaps are completely normal, especially for mothers. Here's how to handle them: 1) Be honest but brief about the gap 2) Focus on any skills you developed during the break 3) Highlight volunteer work, courses, or freelance projects 4) Use a functional resume format to emphasize skills over timeline 5) Consider a brief explanation like 'Career break for family responsibilities'",
    
    "flexible_careers": "Great flexible career options for moms include: Remote roles in tech (customer support, content writing, virtual assistance), Freelance work (graphic design, consulting, tutoring), Part-time positions in education or healthcare, E-commerce and online businesses, Project-based consulting in your field of expertise.",
    
    "maternity_rights": "Maternity rights vary by country: UK: Up to 52 weeks maternity leave, statutory maternity pay. US: 12 weeks unpaid leave (FMLA), varies by state. Canada: Up to 18 months parental leave with benefits. Australia: 18 weeks paid parental leave. EU: Minimum 14 weeks with at least 70% pay. Always check your specific location's laws.",
    
    "returning_confidence": "It's normal to feel anxious about returning to work. Here are some tips: 1) Start with networking events or online communities 2) Take a refresher course in your field 3) Consider contract or part-time work first 4) Update your skills with online courses 5) Practice interviewing with friends 6) Remember that your parenting experience has given you valuable skills like multitasking, patience, and organization."
}

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        return pdfminer.high_level.extract_text(file)
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")

def extract_skills_from_cv(cv_text: str) -> List[str]:
    """Extract skills from CV text using OpenAI"""
    try:
        messages = [
            {"role": "system", "content": "You are a skill extraction expert. Extract key skills, technologies, and competencies from the CV text. Return only a comma-separated list of skills, no explanations."},
            {"role": "user", "content": f"Extract skills from this CV:\n{cv_text[:2000]}"}  # Limit text length
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3,
            max_tokens=200
        )
        
        skills_text = response.choices[0].message.content.strip()
        skills = [skill.strip() for skill in skills_text.split(',') if skill.strip()]
        return skills[:10]  # Return top 10 skills
        
    except Exception as e:
        logger.error(f"Error extracting skills: {e}")
        return []

def get_ai_response(messages: List[Dict], max_tokens: int = 500) -> str:
    """Get response from OpenAI with error handling"""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again later."

@app.post("/welcome")
async def welcome_user(request: Request, token: str = Depends(verify_token)):
    """Welcome and onboard new users"""
    try:
        data = await request.json()
    except Exception:
        data = {}

    user_name = data.get("name", "there")
    returning_user = data.get("returning", False)

    if returning_user:
        welcome_message = f"Welcome back, {user_name}! How can I help you today?"
    else:
        welcome_message = (
            f"Hi {user_name}! 👋 Welcome to Motherboard - your career companion for navigating work and motherhood.\n\n"
            "I'm here to help you with:\n"
            "🔍 **Job Search** - Find flexible, remote, and part-time opportunities\n"
            "📄 **CV Review** - Get personalized feedback on your resume\n"
            "🎯 **Career Guidance** - Explore new career paths and opportunities\n"
            "📧 **Job Alerts** - Subscribe to receive relevant job notifications\n"
            "❓ **Support** - Get answers about returning to work, CV gaps, and more\n\n"
            "What would you like to start with today?"
        )

    return JSONResponse({
        "response": welcome_message,
        "suggestions": [
            "Upload my CV for review",
            "Search for jobs",
            "I need career guidance",
            "Tell me about flexible work options"
        ]
    })

@app.post("/cv-tips")
async def get_cv_tips(file: UploadFile = File(...), token: str = Depends(verify_token)):
    try:
        if file.filename.endswith(".pdf"):
            content = extract_text_from_pdf(file.file)
        elif file.filename.endswith(".docx"):
            content = extract_text_from_docx(file.file)
        else:
            return {"error": "Unsupported file type"}

        messages = [
            {"role": "system", "content": "You are a professional career coach helping users improve their CVs."},
            {"role": "user", "content": f"Here's my resume:\n{content}\n\nHow can I improve it?"}
        ]

        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )

        return {"response": response.choices[0].message.content.strip()}
    
    except Exception as e:
        return {"error": str(e)}
    
    
# Updated main.py - Job Search Endpoint

@app.post("/search-jobs")
async def search_jobs(request: Request, token: str = Depends(verify_token)):
    try:
        data = await request.json()
        user_query = data.get("query", "")
        logger.info(f"Received search query: {user_query}")
        
        # Load jobs first to verify data
        jobs = get_all_jobs()
        logger.info(f"Total jobs available: {len(jobs)}")
        
        matches = find_jobs_from_sentence(user_query)
        
        # Remove embedding field from matches
        clean_matches = []
        for job in matches:
            job_dict = dict(job)
            if 'embedding' in job_dict:
                del job_dict['embedding']
            clean_matches.append(job_dict)
            
        logger.info(f"Found {len(clean_matches)} matches")
        return {"matches": clean_matches}
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {"error": "Search failed", "details": str(e)}
    
    
@app.post("/career-path")
async def suggest_career_path(request: Request, token: str = Depends(verify_token)):
    """Provide career guidance from a single free-form query"""
    try:
        data = await request.json()
        query = data.get("query", "").strip()

        if not query:
            return JSONResponse({
                "response": "Please tell me about your background, experience, interests, and work preferences in one message. For example:\n"
                            "'I'm a former marketing manager with 10 years' experience, looking for remote work in sustainability.'"
            })

        # Generate career advice
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a supportive career coach specializing in helping mothers find fulfilling work. "
                    "From the provided query, suggest 2–3 specific, actionable career paths. "
                    "Include practical next steps and be encouraging."
                )
            },
            {
                "role": "user",
                "content": query
            }
        ]
        career_advice = get_ai_response(messages, max_tokens=600)

        # Use the full query to find relevant jobs
        relevant_jobs = find_jobs_from_sentence(query)
        clean_jobs = []
        for job in relevant_jobs[:3]:
            job_dict = dict(job)
            job_dict.pop("embedding", None)
            clean_jobs.append(job_dict)

        return JSONResponse({
            "response": career_advice,
            "relevant_jobs": clean_jobs,
            "next_steps": [
                "Explore recommended jobs",
                "Subscribe for job alerts",
                "Upload your CV for personalized feedback"
            ]
        })

    except Exception as e:
        logger.error(f"Career path error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, I couldn't provide career guidance right now."}
        )


@app.post("/subscribe")
async def subscribe_user(request: Request, token: str = Depends(verify_token)):
    """Subscribe user for job alerts"""
    try:
        data = await request.json()
        email = data.get("email", "").strip()
        interests = data.get("interests", "Not specified")
        job_types = data.get("job_types", [])
        name = data.get("name", "")

        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return JSONResponse(
                status_code=400,
                content={"error": "Please provide a valid email address."}
            )

        try:
            spreadsheet_id = os.getenv("SPREADSHEET_ID")
            spreadsheet = gs_client.open_by_key(spreadsheet_id)
            sheet = spreadsheet.worksheet("Subscribers")
            
            # Add subscriber data
            sheet.append_row([
                datetime.now().isoformat(),
                name,
                email,
                interests,
                ", ".join(job_types) if job_types else "All types"
            ])

            return JSONResponse({
                "message": f"Successfully subscribed! You'll receive job alerts matching your interests at {email}.",
                "response": "Thank you for subscribing! 🎉 You'll be the first to know about new opportunities that match your preferences.",
                "next_steps": [
                    "Upload your CV for better matches",
                    "Explore current job listings",
                    "Get career guidance"
                ]
            })

        except Exception as e:
            logger.error(f"Subscription error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Sorry, there was an issue with your subscription. Please try again."}
            )

    except Exception as e:
        logger.error(f"Subscribe endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, subscription failed. Please try again."}
        )

@app.post("/faq")
async def handle_faq(request: Request, token: str = Depends(verify_token)):
    """Handle FAQ and general support questions"""
    try:
        data = await request.json()
        question = data.get("question", "").strip().lower()

        # Check for FAQ matches
        faq_response = None
        for key, answer in FAQ_DATA.items():
            if key.replace("_", " ") in question or any(keyword in question for keyword in key.split("_")):
                faq_response = answer
                break

        if faq_response:
            return JSONResponse({
                "response": faq_response,
                "type": "faq",
                "related_topics": list(FAQ_DATA.keys())
            })

        # Use AI for other questions
        messages = [
            {
                "role": "system", 
                "content": "You are a supportive career advisor for mothers returning to work. Provide helpful, encouraging advice about career challenges, work-life balance, and professional development. Keep responses concise and actionable."
            },
            {
                "role": "user", 
                "content": f"Question: {question}"
            }
        ]

        ai_response = get_ai_response(messages, max_tokens=400)

        return JSONResponse({
            "response": ai_response,
            "type": "general",
            "suggestions": [
                "Upload your CV for feedback",
                "Search for flexible jobs",
                "Subscribe for job alerts"
            ]
        })

    except Exception as e:
        logger.error(f"FAQ error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sorry, I couldn't answer your question right now. Please try again."}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/test-auth")
async def test_auth(token: str = Depends(verify_token)):
    return {"message": "Authentication successful"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)