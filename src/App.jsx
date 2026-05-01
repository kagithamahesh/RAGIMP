import { useState } from "react";
import axios from "axios";
import {
  Container,
  Row,
  Col,
  Button,
  Form,
  Card,
  Spinner,
  ListGroup
} from "react-bootstrap";

export default function App() {
  const API = "http://localhost:8000";

  const [files, setFiles] = useState([]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const uploadFiles = async () => {
    if (!files.length) return;

    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
      formData.append("files", files[i]);
    }

    setLoading(true);
    await axios.post(`${API}/upload-multiple`, formData);
    setLoading(false);

    alert("Files indexed successfully");
  };

  const askQuestion = async () => {
    if (!question.trim()) return;

    const userMsg = {
      role: "user",
      text: question
    };

    setMessages((prev) => [...prev, userMsg]);

    setLoading(true);

    const res = await axios.post(`${API}/ask`, {
      question: question
    });

    const aiMsg = {
      role: "assistant",
      text: res.data.answer,
      sources: res.data.sources
    };

    setMessages((prev) => [...prev, aiMsg]);

    setQuestion("");
    setLoading(false);
  };

  return (
    <Container fluid className="p-0">
      <Row className="g-0">

        {/* Sidebar */}
        <Col md={3} className="bg-dark p-4 border-end vh-100">
          <h3>📄 RAG Assistant</h3>

          <Form.Group className="mb-3 mt-4">
            <Form.Label>Upload Files</Form.Label>
            <Form.Control
              type="file"
              multiple
              onChange={(e) => setFiles(e.target.files)}
            />
          </Form.Group>

          <Button
            variant="primary"
            className="w-100 mb-3"
            onClick={uploadFiles}
          >
            Upload
          </Button>

          <Card bg="secondary" text="light">
            <Card.Body>
              Multi-file RAG using React + Bootstrap
            </Card.Body>
          </Card>
        </Col>

        {/* Main Chat */}
        <Col md={9} className="d-flex flex-column vh-100">

          {/* Chat Messages */}
          <div className="chat-scroll p-4">

            {messages.map((msg, index) => (
              <div key={index} className="mb-4">

                <div
                  className={
                    msg.role === "user"
                      ? "user-bubble"
                      : "ai-bubble"
                  }
                >
                  {msg.text}
                </div>

                {msg.sources && (
                  <ListGroup className="mt-2">
                    {msg.sources.map((src, i) => (
                      <ListGroup.Item key={i}>
                        📄 {src}
                      </ListGroup.Item>
                    ))}
                  </ListGroup>
                )}
              </div>
            ))}

            {loading && (
              <Spinner animation="border" />
            )}

          </div>

          {/* Input */}
          <div className="p-3 border-top bg-dark">
            <Row>
              <Col md={10}>
                <Form.Control
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask anything about your files..."
                />
              </Col>

              <Col md={2}>
                <Button
                  variant="success"
                  className="w-100"
                  onClick={askQuestion}
                >
                  Send
                </Button>
              </Col>
            </Row>
          </div>

        </Col>
      </Row>
    </Container>
  );
}