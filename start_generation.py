#!/usr/bin/env python3

import os
import sys
import json
import asyncio
from pathlib import Path
from flask import Flask, request, jsonify
from openai import AsyncOpenAI

app = Flask(__name__)

# Global variables for generation components
client = None

async def init_generation():
    """Initialize generation components"""
    global client
    
    # Initialize OpenAI client for generation
    client = AsyncOpenAI(
        api_key="sk-9a480cd1ba2c4748af4c33becde8bd5a",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    print("✓ Generation service initialized successfully")
    return True

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """Chat completions endpoint compatible with OpenAI API"""
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        model = data.get('model', 'qwen-turbo')
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        
        if not messages:
            return jsonify({'error': 'Messages are required'}), 400
        
        # Call OpenAI API
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        ))
        loop.close()
        
        # Return response in OpenAI format
        return jsonify({
            'id': response.id,
            'object': 'chat.completion',
            'created': response.created,
            'model': response.model,
            'choices': [{
                'index': choice.index,
                'message': {
                    'role': choice.message.role,
                    'content': choice.message.content
                },
                'finish_reason': choice.finish_reason
            } for choice in response.choices],
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models endpoint"""
    return jsonify({
        'object': 'list',
        'data': [
            {
                'id': 'qwen-turbo',
                'object': 'model',
                'created': 1677610602,
                'owned_by': 'alibaba'
            },
            {
                'id': 'qwen-plus',
                'object': 'model', 
                'created': 1677610602,
                'owned_by': 'alibaba'
            },
            {
                'id': 'qwen-max',
                'object': 'model',
                'created': 1677610602,
                'owned_by': 'alibaba'
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'generation'})

if __name__ == '__main__':
    # Initialize generation service in async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    success = loop.run_until_complete(init_generation())
    if not success:
        sys.exit(1)
    
    print("Starting generation HTTP service on port 8000...")
    app.run(host='0.0.0.0', port=8000, debug=False)