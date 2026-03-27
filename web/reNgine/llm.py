import openai
import os
import re
import requests

from dashboard.models import OllamaSettings
from reNgine.common_func import get_open_ai_key, parse_llm_vulnerability_report
from reNgine.definitions import (
    ATTACK_SUGGESTION_GPT_SYSTEM_PROMPT,
    OPENCLAW_CHAT_COMPLETION_PATH,
    OPENCLAW_INSTANCE,
    VULNERABILITY_DESCRIPTION_SYSTEM_MESSAGE,
)


class LLMVulnerabilityReportGenerator:

    def __init__(self, logger):
        selected_model = OllamaSettings.objects.first()
        self.model_name = selected_model.selected_model if selected_model else 'openai:gpt-3.5-turbo'
        self.openai_api_key = None
        self.logger = logger

    def _is_openclaw_model(self):
        return self.model_name.startswith('openclaw:')

    def _resolve_openai_model_name(self):
        if self.model_name.startswith('openai:'):
            return self.model_name.split(':', 1)[1]
        if self.model_name.startswith('openclaw:'):
            return os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-3.5-turbo')
        return self.model_name

    def _get_openclaw_model_name(self):
        if self._is_openclaw_model():
            return self.model_name.split(':', 1)[1]
        return os.getenv('OPENCLAW_MODEL', 'openclaw-default')

    def _invoke_openclaw(self, system_prompt, user_prompt):
        base_url = os.getenv('OPENCLAW_API_BASE', OPENCLAW_INSTANCE).rstrip('/')
        endpoint = f'{base_url}{OPENCLAW_CHAT_COMPLETION_PATH}'
        api_key = os.getenv('OPENCLAW_API_KEY') or get_open_ai_key()
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        payload = {
            'model': self._get_openclaw_model_name(),
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            'temperature': 0.2
        }
        self.logger.info(f'Using OpenClaw API for model {payload["model"]}')
        response = requests.post(
            endpoint,
            json=payload,
            headers=headers,
            timeout=int(os.getenv('OPENCLAW_TIMEOUT', '120'))
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']

    def _invoke_openai(self, system_prompt, user_prompt):
        model_name = self._resolve_openai_model_name()
        self.logger.info(f'Using OpenAI API for model {model_name}')
        openai_api_key = get_open_ai_key()
        if not openai_api_key:
            return None, 'OpenAI API Key not set'
        openai.api_key = openai_api_key
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['choices'][0]['message']['content'], None

    def get_vulnerability_description(self, description):
        self.logger.info(f"Generating Vulnerability Description for: {description}")
        try:
            prompt = re.sub(r'\t', '', VULNERABILITY_DESCRIPTION_SYSTEM_MESSAGE)
            if self._is_openclaw_model():
                response_content = self._invoke_openclaw(prompt, description)
            else:
                response_content, error = self._invoke_openai(prompt, description)
                if error:
                    return {
                        'status': False,
                        'error': error
                    }
        except Exception as e:
            return {
                'status': False,
                'error': str(e)
            }

        response = parse_llm_vulnerability_report(response_content)

        if not response:
            return {
                'status': False,
                'error': 'Failed to parse LLM response'
            }

        return {
            'status': True,
            'description': response.get('description', ''),
            'impact': response.get('impact', ''),
            'remediation': response.get('remediation', ''),
            'references': response.get('references', []),
        }


class LLMAttackSuggestionGenerator:

    def __init__(self, logger):
        self.logger = logger
        self._report_generator = LLMVulnerabilityReportGenerator(logger)

    def get_attack_suggestion(self, user_input):
        try:
            prompt = re.sub(r'\t', '', ATTACK_SUGGESTION_GPT_SYSTEM_PROMPT)
            if self._report_generator._is_openclaw_model():
                response_content = self._report_generator._invoke_openclaw(prompt, user_input)
            else:
                response_content, error = self._report_generator._invoke_openai(prompt, user_input)
                if error:
                    return {
                        'status': False,
                        'error': error,
                        'input': user_input
                    }
        except Exception as e:
            return {
                'status': False,
                'error': str(e),
                'input': user_input
            }
        return {
            'status': True,
            'description': response_content,
            'input': user_input
        }
