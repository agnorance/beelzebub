package plugins

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/go-resty/resty/v2"
	"github.com/mariocandela/beelzebub/v3/tracer"
	log "github.com/sirupsen/logrus"
	"os"
	"regexp"
	"strings"
	"context"
	"time"
	"sync"   
)

const (
	systemPromptVirtualizeLinuxTerminal = `You will act as an Ubuntu Linux terminal. 
											The user will type commands, and you are to reply with what the terminal should show. 
											Your responses must be contained within a single code block. 
											Do not provide note. Do not provide explanations or type commands unless explicitly instructed by the user. 
											Your entire response/output is going to consist of a simple text with \n for new line, 
											and you will NOT wrap it within string md markers`
	systemPromptVirtualizeHTTPServer    = `You will act as an unsecure HTTP Server with multiple vulnerability like aws and git credentials stored into root http directory. 
											The user will send HTTP requests and you are to reply with what the server should show.
											Do not provide explanations or type commands unless explicitly instructed by the user.`
	LLMPluginName                       = "LLMHoneypot"
	openAIEndpoint                      = "https://api.openai.com/v1/chat/completions"
	ollamaEndpoint                      = "http://localhost:11434/api/chat"
	maxHistorySize     = 50                    // Prevent unbounded memory growth
	defaultTimeout     = 30 * time.Second      // API call timeout
	defaultRetryCount  = 3                     // Number of retries
	maxCommandLength   = 10000                 // Prevent excessively long commands
)

// LLMHoneypot simulates network services (SSH, HTTP) using Large Language Models
// to create realistic honeypot interactions for security research and threat detection.
type LLMHoneypot struct {
	// Histories stores the conversation context for multi-turn interactions
	Histories []Message
	
	// OpenAIKey is the API key for OpenAI (can also be set via OPEN_AI_SECRET_KEY env var)
	OpenAIKey string
	
	// client is the HTTP client for making API requests (unexported, injected during init)
	client *resty.Client
	
	// Protocol specifies the service being simulated (SSH, HTTP, etc.)
	Protocol tracer.Protocol
	
	// Provider specifies which LLM provider to use (OpenAI or Ollama)
	Provider LLMProvider
	
	// Model is the specific model name to use (e.g., "gpt-4", "llama2")
	Model string
	
	// Host is the API endpoint (defaults to provider-specific endpoint if empty)
	Host string
	
	// CustomPrompt overrides the default system prompt if provided
	CustomPrompt string

	// Protects Histories from concurrent access
	mu sync.RWMutex
}
type Choice struct {
	Message      Message `json:"message"`
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
}

// Response represents the LLM API response
type Response struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int      `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Message Message  `json:"message"`
	Usage   struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	
	// Add error handling fields
	Error *APIError `json:"error,omitempty"`
}

type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

type Request struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Validate checks if the message is valid
func (m Message) Validate() error {
	if m.Role == "" {
		return errors.New("message role cannot be empty")
	}
	if m.Content == "" {
		return errors.New("message content cannot be empty")
	}
	validRoles := []string{"system", "user", "assistant"}
	for _, valid := range validRoles {
		if m.Role == valid {
			return nil
		}
	}
	return fmt.Errorf("invalid role: %s", m.Role)
}

type Role int

const (
	SYSTEM Role = iota
	USER
	ASSISTANT
)

func (role Role) String() string {
	return [...]string{"system", "user", "assistant"}[role]
}

type LLMProvider int

const (
	Ollama LLMProvider = iota
	OpenAI
)

// String returns the string representation of the provider
func (p LLMProvider) String() string {
	switch p {
	case Ollama:
		return "ollama"
	case OpenAI:
		return "openai"
	default:
		return "unknown"
	}
}

func FromStringToLLMProvider(llmProvider string) (LLMProvider, error) {
	switch strings.ToLower(llmProvider) {
	case "ollama":
		return Ollama, nil
	case "openai":
		return OpenAI, nil
	default:
		return -1, fmt.Errorf("provider %s not found, valid providers: ollama, openai", llmProvider)
	}
}

// InitLLMHoneypot initializes a new LLMHoneypot instance with proper configuration
func InitLLMHoneypot(config *LLMHoneypot) (*LLMHoneypot, error) {
	// Validate configuration
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	
	// Configure HTTP client with timeouts and retries
	config.client = resty.New().
		SetTimeout(defaultTimeout).
		SetRetryCount(defaultRetryCount).
		SetRetryWaitTime(1 * time.Second).
		SetRetryMaxWaitTime(5 * time.Second)

	// Environment variable takes precedence over config
	if envKey := os.Getenv("OPEN_AI_SECRET_KEY"); envKey != "" {
		config.OpenAIKey = envKey
	}
	
	// Validate provider-specific requirements
	if config.Provider == OpenAI && config.OpenAIKey == "" {
		return nil, errors.New("OpenAI provider requires API key (set OPEN_AI_SECRET_KEY or provide OpenAIKey)")
	}

	return config, nil
}

// Validate checks if the LLMHoneypot configuration is valid
func (llmHoneypot *LLMHoneypot) Validate() error {
	if llmHoneypot.Model == "" {
		return errors.New("model must be specified")
	}
	
	if llmHoneypot.Protocol != tracer.SSH && llmHoneypot.Protocol != tracer.HTTP {
		return fmt.Errorf("unsupported protocol: %v", llmHoneypot.Protocol)
	}
	
	return nil
}

// buildPrompt constructs the message array for the LLM API call
func (llmHoneypot *LLMHoneypot) buildPrompt(command string) ([]Message, error) {
	// Validate command input
	if command == "" {
		return nil, errors.New("command cannot be empty")
	}
	
	if len(command) > maxCommandLength {
		return nil, fmt.Errorf("command exceeds maximum length of %d characters", maxCommandLength)
	}
	
	var messages []Message
	var prompt string

	switch llmHoneypot.Protocol {
	case tracer.SSH:
		prompt = systemPromptVirtualizeLinuxTerminal
		if llmHoneypot.CustomPrompt != "" {
			prompt = llmHoneypot.CustomPrompt
		}
	
		// System prompt
		messages = append(messages, Message{
			Role:    SYSTEM.String(),
			Content: prompt,
		})
	
		// Add initial example to set context
		messages = append(messages, Message{
			Role:    USER.String(),
			Content: "pwd",
		})
		messages = append(messages, Message{
			Role:    ASSISTANT.String(),
			Content: "/home/user",
		})
	
		// Add conversation history with thread safety
		llmHoneypot.mu.RLock()
		messages = append(messages, llmHoneypot.Histories...)
		llmHoneypot.mu.RUnlock()
		
	case tracer.HTTP:
		prompt = systemPromptVirtualizeHTTPServer
		if llmHoneypot.CustomPrompt != "" {
			prompt = llmHoneypot.CustomPrompt
		}
		
		// System prompt
		messages = append(messages, Message{
			Role:    SYSTEM.String(),
			Content: prompt,
		})
		
		// Add initial example
		messages = append(messages, Message{
			Role:    USER.String(),
			Content: "GET /index.html",
		})
		messages = append(messages, Message{
			Role:    ASSISTANT.String(),
			Content: "<html><body>Hello, World!</body></html>",
		})
		
		// Add conversation history with thread safety
		llmHoneypot.mu.RLock()
		messages = append(messages, llmHoneypot.Histories...)
		llmHoneypot.mu.RUnlock()
		
	default:
		return nil, fmt.Errorf("unsupported protocol: %v", llmHoneypot.Protocol)
	}
	
	// Add the current command
	messages = append(messages, Message{
		Role:    USER.String(),
		Content: command,
	})

	return messages, nil
}

// openAICaller makes an API call to OpenAI's chat completion endpoint
func (llmHoneypot *LLMHoneypot) openAICaller(ctx context.Context, messages []Message) (string, error) {
	// Validate API key
	if llmHoneypot.OpenAIKey == "" {
		return "", errors.New("openAI API key is empty")
	}

	// Determine endpoint
	endpoint := llmHoneypot.Host
	if endpoint == "" {
		endpoint = openAIEndpoint
	}

	// Marshal request
	requestJson, err := json.Marshal(Request{
		Model:    llmHoneypot.Model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	log.WithFields(log.Fields{
		"provider": "openai",
		"model":    llmHoneypot.Model,
		"endpoint": endpoint,
	}).Debug("Sending request to OpenAI")

	// Make API request with context
	response, err := llmHoneypot.client.R().
		SetContext(ctx).
		SetHeader("Content-Type", "application/json").
		SetBody(requestJson).
		SetAuthToken(llmHoneypot.OpenAIKey).
		SetResult(&Response{}).
		Post(endpoint)

	if err != nil {
		return "", fmt.Errorf("openAI API request failed: %w", err)
	}

	// Check HTTP status code
	if response.StatusCode() != 200 {
		return "", fmt.Errorf("OpenAI API returned status %d: %s", response.StatusCode(), response.String())
	}

	// Safely extract response
	result, ok := response.Result().(*Response)
	if !ok || result == nil {
		return "", errors.New("invalid response format from OpenAI API")
	}
	
	// Check for API errors
	if result.Error != nil {
		return "", fmt.Errorf("OpenAI API error: %s (type: %s, code: %s)", 
			result.Error.Message, result.Error.Type, result.Error.Code)
	}

	// Validate choices
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("no choices in OpenAI response: %s", response.String())
	}

	content := result.Choices[0].Message.Content
	log.WithFields(log.Fields{
		"provider":       "openai",
		"response_chars": len(content),
	}).Debug("Received response from OpenAI")

	return removeQuotes(content), nil
}

// ollamaCaller makes an API call to Ollama's chat endpoint
func (llmHoneypot *LLMHoneypot) ollamaCaller(ctx context.Context, messages []Message) (string, error) {
	// Determine endpoint
	endpoint := llmHoneypot.Host
	if endpoint == "" {
		endpoint = ollamaEndpoint
	}

	// Marshal request
	requestJson, err := json.Marshal(Request{
		Model:    llmHoneypot.Model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	log.WithFields(log.Fields{
		"provider": "ollama",
		"model":    llmHoneypot.Model,
		"endpoint": endpoint,
	}).Debug("Sending request to Ollama")

	// Make API request with context
	response, err := llmHoneypot.client.R().
		SetContext(ctx).
		SetHeader("Content-Type", "application/json").
		SetBody(requestJson).
		SetResult(&Response{}).
		Post(endpoint)

	if err != nil {
		return "", fmt.Errorf("ollama API request failed: %w", err)
	}

	// Check HTTP status code
	if response.StatusCode() != 200 {
		return "", fmt.Errorf("ollama API returned status %d: %s", response.StatusCode(), response.String())
	}

	// Safely extract response
	result, ok := response.Result().(*Response)
	if !ok || result == nil {
		return "", errors.New("invalid response format from Ollama API")
	}
	
	// Check for API errors
	if result.Error != nil {
		return "", fmt.Errorf("ollama API error: %s (type: %s, code: %s)", 
			result.Error.Message, result.Error.Type, result.Error.Code)
	}

	// Ollama returns message directly, not in choices array
	if result.Message.Content == "" {
		return "", fmt.Errorf("empty message content in Ollama response: %s", response.String())
	}

	content := result.Message.Content
	log.WithFields(log.Fields{
		"provider":       "ollama",
		"response_chars": len(content),
	}).Debug("Received response from Ollama")

	return removeQuotes(content), nil
}

// callLLMAPI makes an API call to the configured LLM provider
func (llmHoneypot *LLMHoneypot) callLLMAPI(ctx context.Context, messages []Message) (string, error) {
	switch llmHoneypot.Provider {
	case OpenAI:
		return llmHoneypot.openAICaller(ctx, messages)
	case Ollama:
		return llmHoneypot.ollamaCaller(ctx, messages)
	default:
		return "", fmt.Errorf("unsupported provider: %s", llmHoneypot.Provider)
	}
}

// ExecuteModel executes a command against the LLM and returns the response
func (llmHoneypot *LLMHoneypot) ExecuteModel(command string) (string, error) {
	return llmHoneypot.ExecuteModelWithContext(context.Background(), command)
}

// ExecuteModelWithContext executes a command with context support for cancellation
func (llmHoneypot *LLMHoneypot) ExecuteModelWithContext(ctx context.Context, command string) (string, error) {
	// Validate client initialization
	if llmHoneypot.client == nil {
		return "", errors.New("LLM client not initialized, call InitLLMHoneypot first")
	}
	
	// Validate command
	if command == "" {
		return "", errors.New("command cannot be empty")
	}
	
	if len(command) > maxCommandLength {
		return "", fmt.Errorf("command exceeds maximum length of %d characters", maxCommandLength)
	}

	// Log the execution
	log.WithFields(log.Fields{
		"protocol": llmHoneypot.Protocol,
		"provider": llmHoneypot.Provider,
		"model":    llmHoneypot.Model,
		"command":  command,
	}).Info("Executing LLM model")

	// Build prompt with conversation history
	prompt, err := llmHoneypot.buildPrompt(command)
	if err != nil {
		return "", fmt.Errorf("failed to build prompt: %w", err)
	}

	// Call the LLM API
	response, err := llmHoneypot.callLLMAPI(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("LLM API call failed: %w", err)
	}
	
	// Add to history with thread safety
	llmHoneypot.AddToHistory(Message{
		Role:    USER.String(),
		Content: command,
	})
	llmHoneypot.AddToHistory(Message{
		Role:    ASSISTANT.String(),
		Content: response,
	})

	return response, nil
}

// AddToHistory adds a message to the conversation history with size limiting
func (llmHoneypot *LLMHoneypot) AddToHistory(msg Message) {
	llmHoneypot.mu.Lock()
	defer llmHoneypot.mu.Unlock()
	
	llmHoneypot.Histories = append(llmHoneypot.Histories, msg)
	
	// Keep only recent history to prevent memory bloat and token limit issues
	if len(llmHoneypot.Histories) > maxHistorySize {
		// Remove oldest messages, keeping the most recent ones
		llmHoneypot.Histories = llmHoneypot.Histories[len(llmHoneypot.Histories)-maxHistorySize:]
	}
}

// ClearHistory clears the conversation history
func (llmHoneypot *LLMHoneypot) ClearHistory() {
	llmHoneypot.mu.Lock()
	defer llmHoneypot.mu.Unlock()
	llmHoneypot.Histories = nil
}

// GetHistorySize returns the current size of conversation history
func (llmHoneypot *LLMHoneypot) GetHistorySize() int {
	llmHoneypot.mu.RLock()
	defer llmHoneypot.mu.RUnlock()
	return len(llmHoneypot.Histories)
}

var (
	// Compile regex once at package level for efficiency
	markdownCodeBlockRegex = regexp.MustCompile("```[a-z]*\\n?|```")
)

// removeQuotes strips markdown code block markers from LLM responses
func removeQuotes(content string) string {
	// Remove markdown code blocks (```bash, ```, etc.)
	content = markdownCodeBlockRegex.ReplaceAllString(content, "")
	
	// Trim any leading/trailing whitespace
	return strings.TrimSpace(content)
}
