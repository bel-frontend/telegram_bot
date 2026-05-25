import { ChatOllama } from '@langchain/ollama';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatOpenAI } from '@langchain/openai';

interface ChatModelOptions {
    apiKey?: string;
    ollamaUrl?: string;
}

export enum Model {
    // OpenAI Models
    GPT4o = 'gpt-4o',
    GPT4o_MINI = 'gpt-4o-mini',
    GPT4o_mini = 'gpt-4o-mini',
    GPT4_1 = 'gpt-4.1',
    GPT4_1_MINI = 'gpt-4.1-mini',
    GPT4_1_NANO = 'gpt-4.1-nano',
    GPT5_5 = 'gpt-5.5',
    GPT5 = 'gpt-5',
    GPT5_MINI = 'gpt-5-mini',
    GPT5_NANO = 'gpt-5-nano',
    GPT5_1 = 'gpt-5.1',
    GPT5_2 = 'gpt-5.2',
    GPT5_4 = 'gpt-5.4',
    GPT5_4_MINI = 'gpt-5.4-mini',
    GPT5_4_NANO = 'gpt-5.4-nano',
    O3_MINI = 'o3-mini',
    O3 = 'o3',
    O4_MINI = 'o4-mini',

    // Ollama / Local Models
    LLM3 = 'llama3.2',
    LLAMA3_3 = 'llama3.3:latest',
    GPT_OSS_20B = 'gpt-oss:20b',
    MISTRAL = 'mistral-small3.1',
    GEMMA3_27B = 'gemma3:27b',
    GEMMA3_12B = 'gemma3:12b',
    GEMMA3_12 = 'gemma3:12b',
    GEMMA4 = 'gemma4:latest',
    GEMMA4_26B = 'gemma4:26b',
    GEMMA4_31B = 'gemma4:31b',
    PHI4 = 'phi4:latest',
    QWEN3 = 'qwen3:32b',
    DEEPSEEK = 'deepseek-r1:32b',

    // Google Models
    GEMINI_PRO = 'gemini-2.5-pro',
}

function openAIModel(
    model: Model | string,
    apiKey: string | undefined,
    extra: Partial<ConstructorParameters<typeof ChatOpenAI>[0]> = {}
): ChatOpenAI {
    return new ChatOpenAI({
        model,
        temperature: 0.7,
        apiKey,
        ...extra,
    });
}

export async function chatModel(
    model: Model | string = Model.LLM3,
    options?: ChatModelOptions
): Promise<ChatOpenAI | ChatOllama | ChatGoogleGenerativeAI> {
    let chatModelInstance: ChatOpenAI | ChatOllama | ChatGoogleGenerativeAI;
    const openAIKey = options?.apiKey || process.env.OPENAI_API_KEY;

    switch (model) {
        case Model.GPT4o:
            chatModelInstance = openAIModel(Model.GPT4o, openAIKey);
            break;
        case Model.GPT5_5:
            chatModelInstance = openAIModel(Model.GPT5_5, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5:
            chatModelInstance = openAIModel(Model.GPT5, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'minimal' },
            });
            break;
        case Model.GPT5_MINI:
            chatModelInstance = openAIModel(Model.GPT5_MINI, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_NANO:
            chatModelInstance = openAIModel(Model.GPT5_NANO, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_4:
            chatModelInstance = openAIModel(Model.GPT5_4, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_4_MINI:
            chatModelInstance = openAIModel(Model.GPT5_4_MINI, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_4_NANO:
            chatModelInstance = openAIModel(Model.GPT5_4_NANO, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT4_1:
            chatModelInstance = openAIModel(Model.GPT4_1, openAIKey);
            break;
        case Model.GPT4_1_MINI:
            chatModelInstance = openAIModel(Model.GPT4_1_MINI, openAIKey);
            break;
        case Model.GPT4_1_NANO:
            chatModelInstance = openAIModel(Model.GPT4_1_NANO, openAIKey);
            break;
        case Model.O3_MINI:
            chatModelInstance = openAIModel(Model.O3_MINI, openAIKey, {
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.O3:
            chatModelInstance = openAIModel(Model.O3, openAIKey, {
                useResponsesApi: true,
                reasoning: { effort: 'medium' },
            });
            break;
        case Model.O4_MINI:
            chatModelInstance = openAIModel(Model.O4_MINI, openAIKey, {
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_1:
            chatModelInstance = openAIModel(Model.GPT5_1, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT5_2:
            chatModelInstance = openAIModel(Model.GPT5_2, openAIKey, {
                temperature: 1,
                useResponsesApi: true,
                reasoning: { effort: 'low' },
            });
            break;
        case Model.GPT4o_MINI:
        case Model.GPT4o_mini:
            chatModelInstance = openAIModel(Model.GPT4o_MINI, openAIKey);
            break;
        case Model.GEMINI_PRO:
            chatModelInstance = new ChatGoogleGenerativeAI({
                model,
                temperature: 0.7,
                apiKey:
                    options?.apiKey ||
                    process.env.GOOGLE_GEMINI_API_KEY ||
                    process.env.GOOGLE_API_KEY,
            });
            break;
        default:
            chatModelInstance = new ChatOllama({
                model,
                baseUrl:
                    options?.ollamaUrl ||
                    process.env.OLLAMA_BASE_URL ||
                    'http://localhost:11434',
            });
    }

    return chatModelInstance;
}
