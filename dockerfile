# 1. Use a slim Python base
FROM python:3.10.19-slim

# 2. Optimized performance variables 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Professional working directory
WORKDIR /app

# 4. STEP CHANGE: Copy only requirements first to enable caching
# This way, Docker only re-runs pip install if requirements.txt changes
COPY requirements.txt .

# 5. Install dependencies without bloating the image
RUN pip install -r requirements.txt

# 6. Copy the code last (it changes the most)
COPY src/api/ .
COPY models/ ./models/
COPY configs/ ./configs/


# 7. Security: Run as a non-root user (Prevents potential exploits)
RUN adduser --disabled-password appuser
USER appuser

# 8. Expose the port
EXPOSE 8000

# 9. Start application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]