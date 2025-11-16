📺 YouTube Video AI Assistant

This project is a YouTube Video AI Assistant that allows users to ask questions about any YouTube video, extract insights, summarize content, and chat with the video using AI models.

The app takes a YouTube link, processes the transcript, and responds using an LLM.

🚀 Features

Extract transcript from any YouTube video

Clean and process the transcript

Ask questions about the video content

Summaries, explanations, and insights

Simple and easy-to-run Python project

📂 Project Structure
Youtube_AI_Assistant/
│
├── app.py
├── chatbot_fixed.py
├── requirements.txt
└── README.md

📦 Installation
1. Clone the repository
```bash   
git clone https://github.com/Arpitbanait/Youtube_video_AI_Assistant.git
cd Youtube_video_AI_Assistant
```
⚙️ Environment Setup

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```


Mac/Linux:

```bash
source venv/bin/activate
```

📥 Install Dependencies
3. Install all required packages

```bash
pip install -r requirements.txt
```

🔑 Create Environment Variables

Inside the project folder, create a file named:

```bash
.env
```

Add your keys:

```bash
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_API_KEY=your_token_here
```

Never commit your .env file to GitHub.
Make sure .gitignore contains:

```bash
.env
```

▶️ Running the Project
4. Run the main app

```bash
Streamlit run app.py
```

📝 Usage

Enter the YouTube video link

Ask any question about the video

The AI assistant will answer based on the transcript

Use it for learning, research, summaries, revision, etc.

🤝 Contributions

Feel free to open issues or submit pull requests to improve the project.
