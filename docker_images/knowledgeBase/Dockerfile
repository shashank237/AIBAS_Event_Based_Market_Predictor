FROM busybox

WORKDIR /knowledge

# Copy trained models into knowledge base
COPY models/ ./models/

# Keep container alive
CMD ["sh", "-c", "echo 'Knowledge Base ready' && sleep infinity"]

