package plugins

import (
	"github.com/go-resty/resty/v2"
	"github.com/jarcoal/httpmock"
	"github.com/mariocandela/beelzebub/v3/tracer"
	"github.com/stretchr/testify/assert"
	"net/http"
	"os"
	"testing"
)

const SystemPromptLen = 4

func TestBuildPromptEmptyHistory(t *testing.T) {
	//Given
	var histories []Message
	command := "pwd"

	honeypot := LLMHoneypot{
		Histories: histories,
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	//When
	prompt, err := honeypot.buildPrompt(command)

	//Then
	assert.Nil(t, err)
	assert.Equal(t, SystemPromptLen, len(prompt))
}

func TestBuildPromptWithHistory(t *testing.T) {
	//Given
	var histories = []Message{
		{
			Role:    "user",
			Content: "cat hello.txt",
		},
	}

	command := "pwd"

	honeypot := LLMHoneypot{
		Histories: histories,
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	//When
	prompt, err := honeypot.buildPrompt(command)

	//Then
	assert.Nil(t, err)
	assert.Equal(t, SystemPromptLen+1, len(prompt))
}

func TestBuildPromptWithCustomPrompt(t *testing.T) {
	//Given
	var histories = []Message{
		{
			Role:    "user",
			Content: "cat hello.txt",
		},
	}

	command := "pwd"

	honeypot := LLMHoneypot{
		Histories:    histories,
		Protocol:     tracer.SSH,
		CustomPrompt: "act as calculator",
		Model:        "test-model",
	}

	//When
	prompt, err := honeypot.buildPrompt(command)

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "act as calculator", prompt[0].Content)
	assert.Equal(t, SYSTEM.String(), prompt[0].Role)
}

func TestBuildPromptEmptyCommand(t *testing.T) {
	//Given
	honeypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	//When
	_, err := honeypot.buildPrompt("")

	//Then
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "command cannot be empty")
}

func TestBuildPromptCommandTooLong(t *testing.T) {
	//Given
	honeypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	// Create a command longer than maxCommandLength
	longCommand := make([]byte, maxCommandLength+1)
	for i := range longCommand {
		longCommand[i] = 'a'
	}

	//When
	_, err := honeypot.buildPrompt(string(longCommand))

	//Then
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "exceeds maximum length")
}

func TestBuildExecuteModelFailValidation(t *testing.T) {
	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "",
		Protocol:  tracer.SSH,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	_, err := InitLLMHoneypot(&llmHoneypot) // Add &
	
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "OpenAI provider requires API key")
}

func TestBuildExecuteModelOpenAISecretKeyFromEnv(t *testing.T) {
	defer os.Unsetenv("OPEN_AI_SECRET_KEY")
	
	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "",
		Protocol:  tracer.SSH,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	os.Setenv("OPEN_AI_SECRET_KEY", "sdjdnklfjndslkjanfk")

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot) // Add &

	assert.Nil(t, err)
	assert.Equal(t, "sdjdnklfjndslkjanfk", openAIGPTVirtualTerminal.OpenAIKey)
}

func TestBuildExecuteModelWithCustomPrompt(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	httpmock.RegisterMatcherResponder("POST", openAIEndpoint,
		httpmock.BodyContainsString("hello world"),
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Choices: []Choice{
					{
						Message: Message{
							Role:    SYSTEM.String(),
							Content: "[default]\nregion = us-west-2\noutput = json",
						},
					},
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories:    make([]Message, 0),
		OpenAIKey:    "sdjdnklfjndslkjanfk",
		Protocol:     tracer.HTTP,
		Model:        "gpt-4o",
		Provider:     OpenAI,
		CustomPrompt: "hello world",
	}

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot) // Add &
	assert.Nil(t, err)
	openAIGPTVirtualTerminal.client = client

	str, err := openAIGPTVirtualTerminal.ExecuteModel("GET /.aws/credentials")

	assert.Nil(t, err)
	assert.Equal(t, "[default]\nregion = us-west-2\noutput = json", str)
}

func TestInitLLMHoneypotFailValidationUnsupportedProtocol(t *testing.T) {
	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "test-key",
		Protocol:  tracer.TCP,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	_, err := InitLLMHoneypot(&llmHoneypot)

	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "unsupported protocol")
}

func TestExecuteModelFailValidationInvalidProvider(t *testing.T) {
	// Given
	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "test-key",
		Protocol:  tracer.SSH,
		Model:     "llama3",
		Provider:  5, // Invalid provider
	}

	honeypot, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err) // Init should succeed
	
	// Override client for testing
	honeypot.client = resty.New()

	//When
	_, err = honeypot.ExecuteModel("ls")

	//Then
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "unsupported provider")
}

func TestExecuteModelSSHWithResultsOpenAI(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", openAIEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Choices: []Choice{
					{
						Message: Message{
							Role:    SYSTEM.String(),
							Content: "prova.txt",
						},
					},
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "sdjdnklfjndslkjanfk",
		Protocol:  tracer.SSH,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	openAIGPTVirtualTerminal.client = client

	//When
	str, err := openAIGPTVirtualTerminal.ExecuteModel("ls")

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "prova.txt", str)
	
	// Verify history was updated
	assert.Equal(t, 2, openAIGPTVirtualTerminal.GetHistorySize())
}

func TestExecuteModelSSHWithResultsOllama(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", ollamaEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Message: Message{
					Role:    SYSTEM.String(),
					Content: "prova.txt",
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "llama3",
		Provider:  Ollama,
	}

	ollamaVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	ollamaVirtualTerminal.client = client

	//When
	str, err := ollamaVirtualTerminal.ExecuteModel("ls")

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "prova.txt", str)
}

func TestExecuteModelSSHWithoutResults(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", openAIEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Choices: []Choice{},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "sdjdnklfjndslkjanfk",
		Protocol:  tracer.SSH,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	openAIGPTVirtualTerminal.client = client

	//When
	_, err = openAIGPTVirtualTerminal.ExecuteModel("ls")

	//Then
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "no choices in openAI response")
}

func TestExecuteModelHTTPWithResults(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", openAIEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Choices: []Choice{
					{
						Message: Message{
							Role:    SYSTEM.String(),
							Content: "[default]\nregion = us-west-2\noutput = json",
						},
					},
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "sdjdnklfjndslkjanfk",
		Protocol:  tracer.HTTP,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	openAIGPTVirtualTerminal.client = client

	//When
	str, err := openAIGPTVirtualTerminal.ExecuteModel("GET /.aws/credentials")

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "[default]\nregion = us-west-2\noutput = json", str)
}

func TestBuildExecuteModelFailValidationStrategyType(t *testing.T) {
	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "test-key",
		Protocol:  tracer.TCP,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	_, err := InitLLMHoneypot(&llmHoneypot) // Add &

	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "unsupported protocol")
}

func TestExecuteModelHTTPWithoutResults(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", openAIEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Choices: []Choice{},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		OpenAIKey: "sdjdnklfjndslkjanfk",
		Protocol:  tracer.HTTP,
		Model:     "gpt-4o",
		Provider:  OpenAI,
	}

	openAIGPTVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	openAIGPTVirtualTerminal.client = client

	//When
	_, err = openAIGPTVirtualTerminal.ExecuteModel("GET /.aws/credentials")

	//Then
	assert.NotNil(t, err)
	assert.Contains(t, err.Error(), "no choices in openAI response")
}

func TestFromString(t *testing.T) {
    model, err := FromStringToLLMProvider("openai")
    assert.Nil(t, err)
    assert.Equal(t, OpenAI, model)
    
    model, err = FromStringToLLMProvider("ollama")
    assert.Nil(t, err)
    assert.Equal(t, Ollama, model)
    
    _, err = FromStringToLLMProvider("beelzebub-model")
    assert.NotNil(t, err)
    assert.Contains(t, err.Error(), "provider beelzebub-model not found")
}

func TestExecuteModelSSHWithoutPlaintextSection(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", ollamaEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Message: Message{
					Role:    SYSTEM.String(),
					Content: "```plaintext\n```\n",
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "llama3",
		Provider:  Ollama,
	}

	ollamaVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	ollamaVirtualTerminal.client = client

	//When
	str, err := ollamaVirtualTerminal.ExecuteModel("ls")

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "", str)
}

func TestExecuteModelSSHWithoutQuotesSection(t *testing.T) {
	client := resty.New()
	httpmock.ActivateNonDefault(client.GetClient())
	defer httpmock.DeactivateAndReset()

	// Given
	httpmock.RegisterResponder("POST", ollamaEndpoint,
		func(req *http.Request) (*http.Response, error) {
			resp, err := httpmock.NewJsonResponse(200, &Response{
				Message: Message{
					Role:    SYSTEM.String(),
					Content: "```\n```\n",
				},
			})
			if err != nil {
				return httpmock.NewStringResponse(500, ""), nil
			}
			return resp, nil
		},
	)

	llmHoneypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "llama3",
		Provider:  Ollama,
	}

	ollamaVirtualTerminal, err := InitLLMHoneypot(&llmHoneypot)
	assert.Nil(t, err)
	ollamaVirtualTerminal.client = client

	//When
	str, err := ollamaVirtualTerminal.ExecuteModel("ls")

	//Then
	assert.Nil(t, err)
	assert.Equal(t, "", str)
}

func TestRemoveQuotes(t *testing.T) {
	plaintext := "```plaintext\n```"
	bash := "```bash\n```"
	onlyQuotes := "```\n```"
	complexText := "```plaintext\ntop - 10:30:48 up 1 day,  4:30,  2 users,  load average: 0.15, 0.10, 0.08\nTasks: 198 total,   1 running, 197 sleeping,   0 stopped,   0 zombie\n```"
	complexText2 := "```\ntop - 15:06:59 up 10 days,  3:17,  1 user,  load average: 0.10, 0.09, 0.08\nTasks: 285 total\n```"

	assert.Equal(t, "", removeQuotes(plaintext))
	assert.Equal(t, "", removeQuotes(bash))
	assert.Equal(t, "", removeQuotes(onlyQuotes))
	assert.Equal(t, "top - 10:30:48 up 1 day,  4:30,  2 users,  load average: 0.15, 0.10, 0.08\nTasks: 198 total,   1 running, 197 sleeping,   0 stopped,   0 zombie", removeQuotes(complexText))
	assert.Equal(t, "top - 15:06:59 up 10 days,  3:17,  1 user,  load average: 0.10, 0.09, 0.08\nTasks: 285 total", removeQuotes(complexText2))
}

// New tests for added functionality

func TestAddToHistory(t *testing.T) {
	honeypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	// Add messages
	honeypot.AddToHistory(Message{Role: "user", Content: "ls"})
	honeypot.AddToHistory(Message{Role: "assistant", Content: "file1.txt"})

	assert.Equal(t, 2, honeypot.GetHistorySize())
}

func TestAddToHistoryMaxSize(t *testing.T) {
	honeypot := LLMHoneypot{
		Histories: make([]Message, 0),
		Protocol:  tracer.SSH,
		Model:     "test-model",
	}

	// Add more than maxHistorySize messages
	for i := 0; i < maxHistorySize+10; i++ {
		honeypot.AddToHistory(Message{
			Role:    "user",
			Content: "test",
		})
	}

	// Should be capped at maxHistorySize
	assert.Equal(t, maxHistorySize, honeypot.GetHistorySize())
}

func TestClearHistory(t *testing.T) {
	honeypot := LLMHoneypot{
		Histories: []Message{
			{Role: "user", Content: "test"},
		},
		Protocol: tracer.SSH,
		Model:    "test-model",
	}

	assert.Equal(t, 1, honeypot.GetHistorySize())
	
	honeypot.ClearHistory()
	
	assert.Equal(t, 0, honeypot.GetHistorySize())
}

func TestValidate(t *testing.T) {
	// Valid config
	honeypot := LLMHoneypot{
		Protocol: tracer.SSH,
		Model:    "gpt-4o",
	}
	assert.Nil(t, honeypot.Validate())

	// Missing model
	honeypot2 := LLMHoneypot{
		Protocol: tracer.SSH,
		Model:    "",
	}
	assert.NotNil(t, honeypot2.Validate())

	// Unsupported protocol
	honeypot3 := LLMHoneypot{
		Protocol: tracer.TCP,
		Model:    "gpt-4o",
	}
	assert.NotNil(t, honeypot3.Validate())
}

func TestMessageValidate(t *testing.T) {
	// Valid message
	msg := Message{Role: "user", Content: "test"}
	assert.Nil(t, msg.Validate())

	// Empty role
	msg2 := Message{Role: "", Content: "test"}
	assert.NotNil(t, msg2.Validate())

	// Empty content
	msg3 := Message{Role: "user", Content: ""}
	assert.NotNil(t, msg3.Validate())

	// Invalid role
	msg4 := Message{Role: "invalid", Content: "test"}
	assert.NotNil(t, msg4.Validate())
}

func TestLLMProviderString(t *testing.T) {
	assert.Equal(t, "openai", OpenAI.String())
	assert.Equal(t, "ollama", Ollama.String())
	assert.Equal(t, "unknown", LLMProvider(999).String())
}