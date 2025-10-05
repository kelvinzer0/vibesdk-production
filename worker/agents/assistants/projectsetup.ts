import { TemplateDetails } from '../../services/sandbox/sandboxTypes';
import { SetupCommandsType, type Blueprint } from '../schemas';
import { createObjectLogger, StructuredLogger } from '../../logger';
import { generalSystemPromptBuilder, PROMPT_UTILS } from '../prompts';
import {
	createAssistantMessage,
	createSystemMessage,
	createUserMessage,
} from '../inferutils/common';
import { executeInference } from '../inferutils/infer';
import Assistant from './assistant';
import { AIModels, InferenceContext } from '../inferutils/config.types';
import { extractCommands } from '../utils/common';

interface GenerateSetupCommandsArgs {
	env: Env;
	agentId: string;
	query: string;
	blueprint: Blueprint;
	template: TemplateDetails;
	inferenceContext: InferenceContext;
}

const SYSTEM_PROMPT = `You are a Senior React Native Engineer specializing in project setup and dependency management. Your task is to analyze project requirements and generate precise installation commands for missing dependencies.`;

const SETUP_USER_PROMPT = `## TASK
Analyze the blueprint and generate exact \`bun add\` commands for missing React Native dependencies. Only suggest packages that are NOT already in the starting template.

## EXAMPLES

**Example 1 - UI Project with animations:**
Blueprint mentions: "A UI-heavy app with custom animations and gestures."
Starting template has: react, react-native
Output:
\`\`\`bash
bun add react-native-ui-lib
bun add react-native-reanimated
bun add react-native-gesture-handler
\`\`\`

**Example 2 - App with SVG and gradients:**
Blueprint mentions: "A visually rich app using SVG icons and gradient backgrounds."
Starting template has: react, react-native, react-native-ui-lib
Output:
\`\`\`bash
bun add react-native-svg
bun add react-native-linear-gradient
\`\`\`

**Example 3 - Already Complete:**
Blueprint mentions: "Simple app with basic components."
Starting template has: react, react-native, react-native-ui-lib
Output:
\`\`\`bash
# No additional dependencies needed
\`\`\`

## RULES
- Use ONLY \`bun add\` commands
- Avoid specifying versions unless necessary for compatibility.
- Check version compatibility (e.g., React Native version requirements).
- Skip dependencies already in the starting template.
- Include common companion packages when needed (e.g., \`@types/*\` for TypeScript projects).
- Focus on blueprint requirements only.

${PROMPT_UTILS.COMMANDS}

<INPUT DATA>
<QUERY>
{{query}}
</QUERY>

<BLUEPRINT>
{{blueprint}}
</BLUEPRINT>

<STARTING TEMPLATE>
{{template}}

These are the only dependencies installed currently
{{dependencies}}
</STARTING TEMPLATE>

You need to make sure **ALL THESE** are installed at the least:
{{blueprintDependencies}}

</INPUT DATA>`;

export class ProjectSetupAssistant extends Assistant<Env> {
	private query: string;
	private logger: StructuredLogger;

	constructor({
		env,
		inferenceContext,
		query,
		blueprint,
		template,
	}: GenerateSetupCommandsArgs) {
		const systemPrompt = createSystemMessage(SYSTEM_PROMPT);
		super(env, inferenceContext, systemPrompt);
		this.save([
			createUserMessage(
				generalSystemPromptBuilder(SETUP_USER_PROMPT, {
					query,
					blueprint,
					templateDetails: template,
					dependencies: template.deps,
				}),
			),
		]);
		this.query = query;
		this.logger = createObjectLogger(this, 'ProjectSetupAssistant');
	}

	async generateSetupCommands(error?: string): Promise<SetupCommandsType> {
		this.logger.info('Generating setup commands', {
			query: this.query,
			queryLength: this.query.length,
		});

		try {
			let userPrompt = createUserMessage(
				`Now please suggest required setup commands for the project, inside markdown code fence`,
			);
			if (error) {
				this.logger.info(`Regenerating setup commands after error: ${error}`);
				userPrompt =
					createUserMessage(`Some of the previous commands you generated might not have worked. Please review these and generate new commands if required, maybe try a different version or correct the name?
If the package simply doesn't exist, please don't suggest it.

${error}`);
				this.logger.info(
					`Regenerating setup commands with new prompt: ${userPrompt.content}`,
				);
			}
			const messages = this.save([userPrompt]);

			const results = await executeInference({
				env: this.env,
				messages,
				agentActionName: 'projectSetup',
				context: this.inferenceContext,
				modelName: error ? AIModels.GEMINI_2_5_FLASH : undefined,
			});
			if (!results || typeof results !== 'string') {
				this.logger.info(`Failed to generate setup commands, results: `, {
					results,
				});
				return { commands: [] };
			}

			this.logger.info(`Generated setup commands: ${results}`);

			this.save([createAssistantMessage(results)]);
			return { commands: extractCommands(results) };
		} catch (error) {
			this.logger.error('Error generating setup commands:', error);
			throw error;
		}
	}
}
