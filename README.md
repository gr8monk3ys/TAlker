# TAlker - AI-Powered Teaching Assistant Bot

An innovative solution for automating teaching assistant tasks using the Piazza API, LangChain, FAISS, and OpenAI. This bot helps teaching assistants manage and respond to student questions efficiently through an intuitive web interface.

## Features

- **Automated Response Generation**: Uses OpenAI and LangChain to generate contextually relevant responses to student questions
- **Piazza Integration**: Seamlessly integrates with Piazza's platform for managing course Q&A
- **Smart Search**: Utilizes FAISS for efficient similarity search and retrieval of relevant information
- **Modern Web Interface**: Built with Streamlit for a clean, responsive user experience
- **Secure Authentication**: Implements user authentication to protect sensitive course data

## Prerequisites

- Python 3.9 or higher
- Poetry (Python package manager)
- OpenAI API key
- Piazza account credentials

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gr8monk3ys/TAlker.git
   cd TAlker
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your credentials:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PIAZZA_EMAIL=your_piazza_email
   PIAZZA_PASSWORD=your_piazza_password
   PIAZZA_COURSE_ID=your_course_id
   ```

3. Install dependencies using Poetry:
   ```bash
   make setup
   ```

## Usage

1. Start the application:
   ```bash
   make run
   ```

2. For development with auto-reload:
   ```bash
   make dev
   ```

3. Access the web interface at `http://localhost:8501`

## Development

- Format code:
  ```bash
  make format
  ```

- Run linting:
  ```bash
  make lint
  ```

- Run tests:
  ```bash
  make test
  ```

## Project Structure

```
TAlker/
├── src/
│   ├── dashboard/         # Web interface components
│   │   ├── Home.py       # Main Streamlit application
│   │   └── llm.py        # LangChain integration
│   └── piazza_bot/       # Core bot functionality
│       ├── bot.py        # Piazza API integration
│       ├── profile.py    # User profile management
│       └── responses.py  # Response generation
├── data/                 # Data storage
├── tests/               # Test suite
├── pyproject.toml       # Poetry dependencies
└── Makefile            # Development commands
```

## Available Commands

- `make setup`: Install Poetry and project dependencies
- `make run`: Start the Streamlit application
- `make dev`: Start the application in development mode
- `make lint`: Run Pylint for code quality checks
- `make format`: Format code using Black
- `make test`: Run the test suite
- `make clean`: Clean up cache and temporary files

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Format and lint your code (`make format && make lint`)
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

## License

This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Piazza API](https://github.com/hfaran/piazza-api)
- [LangChain](https://github.com/hwchase17/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

[Back to top](#talker---ai-powered-teaching-assistant-bot)
