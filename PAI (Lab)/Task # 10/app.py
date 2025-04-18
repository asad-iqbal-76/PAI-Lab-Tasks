from flask import Flask, render_template, request, session
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = "restaurant_secret_key"
app.permanent_session_lifetime = timedelta(minutes=5)

def get_bot_response(user_input):
    user_input = user_input.lower()
    if "menu" in user_input:
        return {
            "type": "menu",
            "content": {
                "Starters": ["Soup of the Day", "Spring Rolls", "Garlic Bread", "Chicken Wings"],
                "Main Course": ["Grilled Chicken with Rice", "Beef Steak", "Paneer Tikka Masala", "Spaghetti Bolognese"],
                "Desserts": ["Chocolate Lava Cake", "Cheesecake", "Fruit Trifle", "Ice Cream"],
                "Beverages": ["Fresh Lime", "Mint Margarita", "Soft Drinks", "Tea/Coffee"]
            }
        }
    elif "hours" in user_input or "timing" in user_input:
        return "We’re open from 10:00 AM to 11:00 PM every day!"
    elif "special" in user_input:
        return "Today's special: Grilled Chicken Alfredo Pasta served with garlic bread."
    elif "location" in user_input or "address" in user_input:
        return "Bahria Town, Lahore, Pakistan"
    elif "contact" in user_input or "phone" in user_input:
        return "Call us at (042) 1442-8652 or WhatsApp +92 301 8092288"
    elif "book" in user_input or "reservation" in user_input:
        return "You can book a table by calling us or using our website’s reservation system."
    elif "feedback" in user_input:
        return "We'd love your feedback! Tell us how your experience was."
    else:
        return "I'm not sure I understand. Try asking about our menu, hours, specials, or reservations."

@app.route("/", methods=["GET", "POST"])
def index():
    if "chat" not in session:
        session["chat"] = []

    if request.method == "POST":
        user_msg = request.form["message"]
        bot_reply = get_bot_response(user_msg)

        session["chat"].append({
            "sender": "You",
            "message": user_msg,
            "time": datetime.now().strftime("%I:%M %p")
        })

        if isinstance(bot_reply, dict) and bot_reply.get("type") == "menu":
            menu_text = "Here's our menu:\n"
            for category, items in bot_reply["content"].items():
                menu_text += f"\n{category}:\n"
                for item in items:
                    menu_text += f" - {item}\n"
            bot_msg = menu_text.strip()
        else:
            bot_msg = bot_reply

        session["chat"].append({
            "sender": "Bot",
            "message": bot_msg,
            "time": datetime.now().strftime("%I:%M %p")
        })

        session.modified = True

    return render_template("index.html", chat=session["chat"])

if __name__ == "__main__":
    app.run(debug=True)
