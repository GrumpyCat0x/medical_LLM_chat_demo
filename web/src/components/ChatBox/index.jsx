import React, { useState, useRef, useEffect } from "react";
import { Input, Button, List, Avatar, message } from "antd";
import { SendOutlined, UserOutlined, RobotOutlined } from "@ant-design/icons";
import axios from "axios";
import ReactMarkdown from "react-markdown";

import "./styles.scss";

const ChatBox = () => {
  const [messages, setMessages] = useState([]);
  const [inputMsg, setInputMsg] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "auto" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSend = async () => {
    if (!inputMsg.trim()) {
      message.warning("Type something to send!");
      return;
    }

    try {
      const userMessage = {
        content: inputMsg,
        isBot: false,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setInputMsg("");
      setLoading(true);

      const { data } = await axios.post("/chat", {
        input: inputMsg,
      });

      console.log(data)

      const botMessage = {
        content: data,
        isBot: true,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err) {
      message.error(`Failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="message-list">
        <List
          itemLayout="horizontal"
          dataSource={messages}
          renderItem={(item) => (
            <List.Item
              className={item.isBot ? "bot-message" : "user-message"}
              style={{
                justifyContent: item.isBot ? "flex-start" : "flex-end",
                padding: "8px 0",
              }}
            >
              <div className="message-bubble">
                <Avatar
                  className="message-avatar"
                  icon={item.isBot ? <RobotOutlined /> : <UserOutlined />}
                />
                <div className="message-content">
                  <ReactMarkdown
                    components={{
                      code: ({ node, ...props }) => (
                        <code className="markdown-code" {...props} />
                      ),
                    }}
                  >
                    {item.content}
                  </ReactMarkdown>
                </div>
              </div>
            </List.Item>
          )}
        />
        <div ref={messagesEndRef} />
      </div>

      <div className="input-area">
        <Input.TextArea
          value={inputMsg}
          onChange={(e) => setInputMsg(e.target.value)}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Input (Shift+Enter start a new line ...)"
          autoSize={{ minRows: 2, maxRows: 6 }}
          disabled={loading}
          allowClear
        />
        <Button
          className="send-button"
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          loading={loading}
        >
          {loading ? "Generating..." : "Send"}
        </Button>
      </div>
    </div>
  );
};

export default ChatBox;
