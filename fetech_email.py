from dotenv import load_dotenv
load_dotenv()
import json
import datetime
import enum
import os
import sys
import time
import urllib.parse
import re

from google import genai
from google.genai.errors import ClientError
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from nylas import Client
import requests

# Set user name
USER_NAME = "Pranav Ponnouswamy"

class EmailCategory(enum.Enum):
    REPLY = "Reply"
    ARCHIVE = "Archive"
    Unsubscribe = "Unsubscribe"
    Forward = "Forward"

# Initialize the Nylas client with your env variables
nylas = Client(
    os.environ.get('NYLAS_API_KEY'),
    os.environ.get('NYLAS_API_URI')
)

grant_id = os.environ.get("NYLAS_GRANT_ID")
one_week_ago = int((datetime.datetime.now() - datetime.timedelta(days=7)).timestamp())

# Fetch messages from the last 7 days (limit 100 for example)
messages = nylas.messages.list(
    grant_id,
    query_params={
        "limit": 100,
        # "received_after": one_week_ago,
        "in": "INBOX"
    }
)

# Initialize Google GenAI client
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Initialize Qdrant vector database client (in-memory for demo)
q_client = QdrantClient(":memory:")
collection_name = "test_collection"

# Check/create the collection in Qdrant
try:
    q_client.get_collection(collection_name=collection_name)
    print(f"Collection '{collection_name}' already exists.")
except Exception:
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )


def categorize_email(subject):
    """Categorize an email subject using GenAI and Qdrant context."""
    try:
        # Embed the subject text
        subject_embedding = client.models.embed_content(
            model="text-embedding-004",
            contents=subject
        )
        query_vector = subject_embedding.embeddings[0].values

        # Query the vector DB
        results = q_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=3
        ).points

        print("\n=== Similar Previous Emails ===")
        for i, res in enumerate(results, 1):
            p = res.payload
            print(f"\nSimilar Email #{i}:")
            print(f"Subject: {p.get('subject','')}")
            print(f"Action: {p.get('action',{})}")
            print(f"Summary: {p.get('summary','')}\n")
            print("-" * 50)

        # Build context from retrieved emails
        context = ""
        for res in results:
            p = res.payload
            context += f"Email subject: {p.get('subject','')}. Action: {p.get('action',{})}\n"

        # Compose prompt
        prompt_text = (
            f"Context:\n{context}\n---\nCategorize this email subject: {subject}."
        )

        # Request an enum-based response from GenAI
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={
                'response_mime_type': 'text/x.enum',
                'response_schema': EmailCategory,
            },
        )
        return response.text

    except ClientError as e:
        print("ClientError encountered:", e)
        return "ResourceExhausted"


def summarize_email(email_body):
    """Summarize an email body using GenAI."""
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=f"Summarize the following email: {email_body}",
            config={'response_mime_type': 'text/plain'}
        )
        return response.text
    except ClientError as e:
        print("ClientError encountered during summary:", e)
        return "Summary not available."


def round_to_5_minutes(dt: datetime.datetime) -> datetime.datetime:
    """
    Rounds a datetime down to the nearest 5-minute mark.
    Nylas availability endpoints require times to be multiples of 5 minutes.
    """
    total_minutes = int(dt.timestamp() // 60)
    # round down to nearest multiple of 5
    new_minutes = (total_minutes // 5) * 5
    new_timestamp = new_minutes * 60
    return datetime.datetime.fromtimestamp(new_timestamp)


def get_calendar_availability(start_time=None, days_ahead=7):
    """
    Get free times for the next week using Nylas 'calendars.get_availability()'.
    Returns times in a human-readable format.
    """
    print("\n=== Checking Calendar ===")
    if not start_time:
        start_time = datetime.datetime.now()
    end_time = start_time + datetime.timedelta(days=days_ahead)
    
    print(f"Looking for availability: {start_time.strftime('%B %d')} - {end_time.strftime('%B %d')}")

    # Round both to nearest 5-minute increments
    start_time = round_to_5_minutes(start_time)
    end_time = round_to_5_minutes(end_time)

    try:
        user_email = os.environ.get("NYLAS_USER_EMAIL")
        if not user_email:
            print("Error: NYLAS_USER_EMAIL not set in environment")
            return []
        
        # Get availability in 30-minute increments
        request_body = {
            "start_time": int(start_time.timestamp()),
            "end_time": int(end_time.timestamp()),
            "duration_minutes": 30,
            "participants": [{"email": user_email}]
        }
        
        availability = nylas.calendars.get_availability(request_body=request_body)
        print(f"Found {len(availability[0].time_slots)} potential time slots")

        free_slots = []
        if hasattr(availability[0], 'time_slots'):
            for slot in availability[0].time_slots:
                slot_start = datetime.datetime.fromtimestamp(slot.start_time)
                slot_end = datetime.datetime.fromtimestamp(slot.end_time)

                # Skip weekends
                if slot_start.weekday() >= 5:
                    continue

                # Clamp to 9:00–17:00
                day_start = slot_start.replace(hour=9, minute=0, second=0, microsecond=0)
                day_end = slot_start.replace(hour=17, minute=0, second=0, microsecond=0)

                actual_start = max(slot_start, day_start)
                actual_end = min(slot_end, day_end)

                if actual_end > actual_start:
                    duration_hrs = (actual_end - actual_start).total_seconds() / 3600.0
                    # Keep slots >= 30 minutes
                    if duration_hrs >= 0.5:
                        slot_info = {
                            'start': actual_start,
                            'end': actual_end,
                            'duration': duration_hrs,
                            'human_readable': f"{actual_start.strftime('%A, %B %d at %I:%M %p')} - {actual_end.strftime('%I:%M %p')}"
                        }
                        free_slots.append(slot_info)

        # Sort by start time and get random selection if we have more than 10 slots
        free_slots.sort(key=lambda x: x['start'])
        if len(free_slots) > 10:
            import random
            free_slots = random.sample(free_slots, 10)
            free_slots.sort(key=lambda x: x['start'])  # Re-sort after sampling
        
        print(f"\nProcessed {len(free_slots)} available slots:")
        for slot in free_slots[:3]:  # Show just first 3 for brevity
            print(f"→ {slot['human_readable']}")
        if len(free_slots) > 3:
            print(f"  (and {len(free_slots)-3} more...)")
        
        return free_slots

    except Exception as e:
        print(f"Error getting calendar availability: {str(e)}")
        import traceback
        print("Full traceback:")
        print(traceback.format_exc())
        return []


def format_available_times(free_slots, limit=10):
    """Format available time slots into a human-readable string."""
    print("\n=== Formatting Times ===")
    if not free_slots:
        return "I don't have any availability in the next week. Could we look at times after that?"
    
    formatted_slots = []
    for slot in free_slots[:limit]:
        duration_str = ""
        if slot['duration'] >= 1:
            duration_str = f" ({int(slot['duration'])} hour{'s' if slot['duration'] > 1 else ''})"
        elif slot['duration'] >= 0.5:
            duration_str = f" ({int(slot['duration'] * 60)} minutes)"
            
        formatted_slot = f"{slot['human_readable']}{duration_str}"
        formatted_slots.append(formatted_slot)
    
    result = (
        "Here are my available time slots for the next week:\n" +
        "\n".join(f"- {slot}" for slot in formatted_slots)
    )
    return result


def is_meeting_request(email_body):
    """Detect if an email is a meeting request."""
    meeting_keywords = [
        r'\b(meet|meeting|discuss|chat|catch up|sync|connect)\b',
        r'\b(availability|schedule|time|slot)\b',
        r'when.*(?:free|available)',
        r'(?:can|could).*(?:meet|talk)',
        r'let(?:\')?s.*(?:meet|talk|discuss)',
    ]
    
    email_lower = email_body.lower()
    return any(re.search(pattern, email_lower) for pattern in meeting_keywords)


def analyze_email_intent(email_body):
    """Analyze the email to determine if it requires a calendar-based response."""
    print("\n=== Analyzing Email Intent ===")
    
    prompt_text = (
        "Analyze this email and determine if it requires scheduling a meeting or suggesting available times.\n\n"
        "Email:\n"
        f"{email_body}\n\n"
        "Instructions:\n"
        "1. Determine if the sender is explicitly or implicitly requesting a meeting, call, or any form of synchronous communication\n"
        "2. Consider phrases like 'let's meet', 'can we discuss', 'when are you free', 'your availability', etc.\n"
        "3. Distinguish between actual meeting requests and rhetorical phrases like 'we should catch up sometime'\n"
        "4. Return ONLY 'yes' if a meeting response is needed, or 'no' if a regular reply is sufficient\n"
        "5. If unsure, default to 'no'\n\n"
        "Response (yes/no):"
    )
    
    try:
        print("Analyzing if calendar response needed...")
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={'response_mime_type': 'text/plain'}
        )
        result = response.text.strip().lower()
        needs_calendar = result == 'yes'
        print(f"→ Calendar response {'needed' if needs_calendar else 'not needed'}")
        return needs_calendar
    except Exception as e:
        print(f"Error analyzing email intent: {e}")
        return False


def suggest_reply(email_body):
    """Suggest a reply for an email using GenAI and similar email context from Qdrant."""
    try:
        # First analyze if this needs a calendar-based response
        if analyze_email_intent(email_body):
            return suggest_meeting_reply(email_body)
            
        # Otherwise, do a standard RAG-based reply
        print("\n=== Generating Regular Reply ===")
        embedding_result = client.models.embed_content(
            model="text-embedding-004",
            contents=email_body
        )
        query_vector = embedding_result.embeddings[0].values

        results = q_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            with_payload=True,
            limit=3
        ).points

        print("\n=== Similar Previous Email Exchanges ===")
        similar_context = ""
        for i, res in enumerate(results, 1):
            p = res.payload
            if p.get('action', {}).get('type') == 'reply':
                print(f"\nSimilar Exchange #{i}:")
                print(f"Subject: {p.get('subject','')}")
                print(f"Summary: {p.get('summary','')}")
                print(f"Previous Reply: {p.get('action', {}).get('reply_text', '')}")
                print("-" * 50)
                similar_context += (
                    f"Email: {p.get('summary', '')}\n"
                    f"Reply: {p.get('action', {}).get('reply_text', '')}\n\n"
                )

        prompt_text = (
            f"You are writing a reply to an email. Here is the context and instructions:\n\n"
            f"Similar past email exchanges:\n{similar_context}\n"
            f"Current email to reply to:\n{email_body}\n\n"
            "Instructions:\n"
            "1. Write a warm and friendly reply\n"
            "2. Keep the tone appropriate to the context\n"
            "3. Format with proper paragraphs and line breaks\n"
            "4. Include appropriate greeting and closing\n"
            "5. Be concise but thorough\n"
            "6. Do NOT include any subject line\n"
            f"7. Sign as {USER_NAME}\n\n"
            "Important: Your reply should NOT start with 'Subject:' or include any email subject line!"
        )
        
        print("\nGenerating smart reply...")
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={'response_mime_type': 'text/plain'}
        )
        result = response.text.strip()
        
        # Double-check no subject line
        if result.lower().startswith('subject:'):
            result = result[result.index('\n')+1:].strip()
            
        print("✓ Reply generated successfully")
        return result
    except ClientError as e:
        print("ClientError encountered during reply suggestion:", e)
        return ""


def suggest_meeting_reply(email_body):
    """Generate a reply for meeting requests with available times."""
    print("\n=== Generating Calendar Reply ===")
    free_slots = get_calendar_availability()
    available_times = format_available_times(free_slots, limit=10)
    
    # Build prompt for meeting reply
    prompt_text = (
        f"You are writing a friendly email reply to a meeting request. Here are 10 available time slots:\n"
        f"{available_times}\n\n"
        f"Original email:\n{email_body}\n\n"
        "Instructions:\n"
        "1. Write a warm and friendly reply\n"
        "2. From the available times above, select the 3 BEST options considering:\n"
        "   - Prefer times during mid-morning or early afternoon\n"
        "   - Avoid back-to-back slots unless they can be combined into a longer meeting\n"
        "   - Try to spread options across different days if possible\n"
        "3. Format the selected times in a clear, easy-to-read way\n"
        "4. Do NOT include any subject line in the reply\n"
        "5. Keep the response concise but warm\n"
        "6. Ask them to confirm which time works best\n"
        f"7. Sign as {USER_NAME}\n\n"
        "Important: Your reply should NOT start with 'Subject:' or include any email subject line!"
    )
    
    try:
        print("Generating smart reply with selected time slots...")
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt_text,
            config={'response_mime_type': 'text/plain'}
        )
        result = response.text.strip()
        
        # Double-check no subject line
        if result.lower().startswith('subject:'):
            result = result[result.index('\n')+1:].strip()
            
        print("✓ Reply generated successfully")
        return result
    except ClientError as e:
        print(f"Error generating meeting reply: {e}")
        # Fallback to a simple response
        fallback = (
            f"Thank you for your email. {available_times}\n\n"
            f"Please let me know if any of these times work for you.\n\n"
            f"Best regards,\n{USER_NAME}"
        )
        print("! Using fallback reply")
        return fallback


def find_unsubscribe_link(email_body, headers):
    """Find unsubscribe link from email headers or body."""
    if headers:
        for header in headers:
            if header.get("name", "").lower() == "list-unsubscribe":
                value = header.get("value", "")
                if "<" in value and ">" in value:
                    url = value[value.find("<")+1:value.find(">")]
                    if url.startswith("http"):
                        return url, "header"
                    elif url.startswith("mailto:"):
                        return url, "mailto"
    
    patterns = [
        r'https?://[^\s<>"]+?/unsubscribe[^\s<>"]*',
        r'https?://[^\s<>"]+?/opt-?out[^\s<>"]*',
        r'https?://[^\s<>"]+?/remove[^\s<>"]*',
        r'mailto:[^\s<>"]+?\?subject=(?:unsubscribe|remove|optout)[^\s<>"]*'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, email_body, re.IGNORECASE)
        if matches:
            return matches[0], "body"
    return None, None


# Optional categories
categories = {
    "Work": [],
    "Personal": [],
    "Spam": [],
    "Newsletter": [],
    "Social": []
}

email_points = {}
nylas_emails = {}

# Main loop
for index, message in enumerate(messages[0]):
    email_body = message.body or ""

    # Categorize
    category = categorize_email(email_body)
    summary = summarize_email(email_body)

    # Embedding
    embedding_result = client.models.embed_content(
        model="text-embedding-004",
        contents=email_body
    )

    sender = ""
    if hasattr(message, "from_") and message.from_:
        sender = message.from_[0].get("email", "")

    recipients_raw = getattr(message, "to", [])
    recipients = [
        r.get("email", r) if isinstance(r, dict) else r for r in recipients_raw
    ]

    payload = {
        "category": category,
        "summary": summary,
        "subject": getattr(message, "subject", ""),
        "date": getattr(message, "date", ""),
        "sender": sender,
        "recipients": recipients,
        "message_id": getattr(message, "id", ""),
        "action": {},
    }

    point = PointStruct(
        id=index,
        vector=embedding_result.embeddings[0].values,
        payload=payload
    )

    email_points[index] = point
    nylas_emails[index] = message
    operation_info = q_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[point],
    )
    print(f"Upsert result for email {index}:", operation_info)

    if payload.get("category") not in categories:
        categories[payload.get("category")] = []
    categories[payload.get("category")].append(message)

    print("\n------------------------")
    print(f"Email [{index}]:")
    print(f"From: {payload.get('sender', '')}")
    print(f"To: {payload.get('recipients', '')}")
    print(f"Subject: {payload.get('subject', '')}")
    print(f"Summary: {payload.get('summary', '')}")
    print(f"Suggested Action: {payload.get('category', 'None')}")

    action = input("Action (archive/reply/forward/skip/unsubscribe): ").strip().lower()
    if action in ["skip", ""]:
        print("Skipping email.")
        continue

    action_payload = {"type": action}
    if action == "reply":
        reply_text = suggest_reply(email_body)
        print("LLM suggested reply:")
        print(reply_text)
        decision = input("Send reply? (yes/edit/skip): ").strip().lower()
        if decision == "edit":
            reply_text = input("Enter your revised reply: ").strip()
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = True
        elif decision == "yes":
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = True
        else:
            print("Reply draft saved but not sent.")
            action_payload["reply_text"] = reply_text
            action_payload["send_reply"] = False
    elif action == "forward":
        forward_to = input("Confirm recipient email for forwarding: ").strip()
        forward_data = {
            "subject": "Fwd: " + (getattr(nylas_email, "subject", "")),
            "body": action_payload.get("reply_text", ""),
            "to": [{"email": forward_to}],
            "attachment_ids": getattr(nylas_email, "file_ids", []),
        }
        forward_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=forward_data)
        print("Email forwarded successfully.", forward_response)
    elif action == "unsubscribe":
        unsubscribe_link, link_type = find_unsubscribe_link(email_body, getattr(message, "headers", None))
        if not unsubscribe_link:
            print("No unsubscribe link found. Archiving email instead.")
            current_folders = getattr(message, "folders", [])
            new_folders = [folder for folder in current_folders if folder != "INBOX"]
            updated_message = nylas.messages.update(
                grant_id,
                message.id,
                request_body={
                    "folders": new_folders,
                    "starred": False,
                    "unread": False,
                }
            )
            print("Email archived:", updated_message)
        else:
            if link_type == "mailto":
                email_addr = unsubscribe_link[7:]
                subject = "Unsubscribe"
                if "?" in email_addr:
                    email_addr, params = email_addr.split("?", 1)
                    for param in params.split("&"):
                        if param.startswith("subject="):
                            subject = urllib.parse.unquote(param[8:])
                
                unsubscribe_data = {
                    "subject": subject,
                    "body": "Please unsubscribe me from this mailing list.",
                    "to": [{"email": email_addr}]
                }
                unsubscribe_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=unsubscribe_data)
                print("Unsubscribe email sent:", unsubscribe_response)
            elif link_type in ("header", "body"):
                try:
                    response = requests.get(unsubscribe_link)
                    if response.status_code == 200:
                        print(f"Successfully accessed unsubscribe link: {unsubscribe_link}")
                    else:
                        print(f"Failed to access unsubscribe link (status code {response.status_code})")
                except Exception as e:
                    print(f"Error accessing unsubscribe link: {e}")
            
            current_folders = getattr(message, "folders", [])
            new_folders = [folder for folder in current_folders if folder != "INBOX"]
            updated_message = nylas.messages.update(
                grant_id,
                message.id,
                request_body={
                    "folders": new_folders,
                    "starred": False,
                    "unread": False,
                }
            )
            print("Email archived after unsubscribe attempt:", updated_message)
    else:
        print("No recognized action performed on Gmail.")

    point.payload["action"] = action_payload
    op_info = q_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=[point],
    )
    print(f"Updated action for email {index}: {op_info}")

    # Reflect action in Gmail
    try:
        nylas_email = nylas_emails.get(index)
        if not nylas_email:
            print("No corresponding Nylas email found.")
            continue

        message_id = getattr(nylas_email, "id", "")
        if action == "archive":
            request_body = {
                "remove_labels": ["INBOX"],
                "starred": False,
                "unread": False,
            }
            current_folders = getattr(nylas_email, "folders", [])
            new_folders = [folder for folder in current_folders if folder != "INBOX"]
            updated_message = nylas.messages.update(
                grant_id,
                nylas_email.id,
                request_body={
                    "folders": new_folders,
                    "starred": False,
                    "unread": False,
                }
            )
            print("Email archived:", updated_message)

        elif action == "reply":
            if action_payload.get("send_reply"):
                reply_text = action_payload.get("reply_text", "")
                thread_id = getattr(nylas_email, "thread_id", None)
                if not thread_id:
                    print("Warning: No thread ID found, email threading may not work properly")
                    thread_id = getattr(nylas_email, "id", "")

                message_id = getattr(nylas_email, "id", "")
                headers = getattr(nylas_email, "headers", []) or []
                references = []
                in_reply_to = None

                for header in headers:
                    name = header.get("name", "").lower()
                    if name == "references":
                        references.extend([ref.strip() for ref in header.get("value", "").split()])
                    elif name == "message-id":
                        in_reply_to = header.get("value")
                
                if not in_reply_to:
                    in_reply_to = f"<{message_id}@nylas>"
                    
                if in_reply_to and in_reply_to not in references:
                    references.append(in_reply_to)

                formatted_reply = reply_text.replace("\n", "<br>")

                reply_data = {
                    "subject": getattr(nylas_email, "subject", ""),
                    "body": formatted_reply,
                    "to": [{"email": nylas_email.from_[0]["email"]}] if hasattr(nylas_email, "from_") and nylas_email.from_ else [],
                    "in_reply_to": in_reply_to,
                    "references": " ".join(references) if references else in_reply_to,
                    "thread_id": thread_id,
                    "reply_to_message_id": message_id
                }

                reply_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=reply_data)
                print("Reply sent successfully.", reply_response)
            else:
                print("Draft saved; reply not sent.")

        elif action == "forward":
            forward_to = input("Confirm recipient email for forwarding: ").strip()
            forward_data = {
                "subject": "Fwd: " + (getattr(nylas_email, "subject", "")),
                "body": action_payload.get("reply_text", ""),
                "to": [{"email": forward_to}],
                "attachment_ids": getattr(nylas_email, "file_ids", []),
            }
            forward_response = nylas.messages.send(os.environ.get("NYLAS_GRANT_ID"), request_body=forward_data)
            print("Email forwarded successfully.", forward_response)
        else:
            print("No recognized action performed on Gmail.")
    except Exception as e:
        print("Error reflecting action in Gmail:", e)

print("All emails processed and upserted.\n")
