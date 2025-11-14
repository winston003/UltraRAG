#!/usr/bin/env python3

import os
import sys
import json
import asyncio
from pathlib import Path
from flask import Flask, request, jsonify
from openai import AsyncOpenAI
import lancedb
import numpy as np

# Add servers to path
sys.path.insert(0, str(Path(__file__).parent / "servers" / "retriever" / "src"))

app = Flask(__name__)

# Global variables for retriever components
client = None
table = None

async def init_retriever():
    """Initialize retriever components - only for retrieval, not building indexes"""
    global client, table
    
    # Initialize OpenAI client for query embedding
    client = AsyncOpenAI(
        api_key="sk-9a480cd1ba2c4748af4c33becde8bd5a",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    # Connect to existing LanceDB - configurable path
    # 使用与构建索引时一致的默认路径
    lancedb_path = os.getenv('LANCEDB_PATH', 'data/lancedb')
    table_name = os.getenv('LANCEDB_TABLE', 'documents')
    
    if os.path.exists(lancedb_path):
        try:
            db = lancedb.connect(lancedb_path)
            if table_name in db.table_names():
                table = db.open_table(table_name)
                print(f"✓ Retriever initialized successfully with {lancedb_path}/{table_name}")
                return True
            else:
                print(f"✗ Table '{table_name}' not found in {lancedb_path}")
                return False
        except Exception as e:
            print(f"✗ Failed to connect to LanceDB: {e}")
            return False
    else:
        print(f"✗ LanceDB path does not exist: {lancedb_path}")
        print("Please run the index building process first, e.g.:")
        print("python process_dashscope.py --input_file your_file.txt --index_type lancedb")
        return False

async def create_embeddings(corpus_path, embedding_path):
    """Create embeddings for corpus"""
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    
    chunks = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            chunks.append(data['contents'])
    
    embeddings = []
    batch_size = 20
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = await client.embeddings.create(
            model="text-embedding-v3",
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Processed {min(i+batch_size, len(chunks))}/{len(chunks)} chunks")
    
    # Save embeddings
    embedding_data = {
        'embeddings': embeddings,
        'texts': chunks
    }
    
    with open(embedding_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_data, f, ensure_ascii=False)
    
    print(f"✓ Embeddings saved to {embedding_path}")

async def create_lancedb_index(embedding_path, lancedb_path):
    """Create LanceDB index"""
    os.makedirs(lancedb_path, exist_ok=True)
    
    with open(embedding_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings'])
    texts = data['texts']
    
    # Create LanceDB table
    db = lancedb.connect(lancedb_path)
    
    table_data = []
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        table_data.append({
            'id': i,
            'text': text,
            'vector': embedding.tolist()
        })
    
    table = db.create_table("documents", table_data, mode="overwrite")
    print(f"✓ LanceDB index created at {lancedb_path}")

@app.route('/search', methods=['POST'])
async def search():
    """Search endpoint"""
    global table
    
    try:
        # 检查表是否已初始化
        if table is None:
            return jsonify({'error': 'Table not initialized'}), 500
        
        data = request.get_json()
        
        # Support both single query and query_list formats
        query_list = data.get('query_list', [])
        single_query = data.get('query', '')
        top_k = data.get('top_k', 5)
        
        # Handle single query format
        if single_query and not query_list:
            query_list = [single_query]
        elif not query_list:
            return jsonify({'error': 'Query or query_list is required'}), 400
        
        # Process all queries
        all_results = []
        
        for query in query_list:
            if not query.strip():
                continue
                
            # Get query embedding
            response = await client.embeddings.create(
                model="text-embedding-v1",
                input=[query]
            )
            query_embedding = response.data[0].embedding
            
            # Search in LanceDB (不使用锁，允许并发查询)
            results = table.search(query_embedding).limit(top_k).to_list()
            
            # Format results as list of strings (matching expected format)
            formatted_results = [result['text'] for result in results]
            all_results.append(formatted_results)
        
        # Return in the format expected by UltraRAG (ret_psg key)
        return jsonify({'ret_psg': all_results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'retriever'})

@app.route('/reload', methods=['POST'])
def reload_table():
    """Reload LanceDB table - hot reload endpoint"""
    global table
    
    try:
        # Get LanceDB path and table name from environment variables
        lancedb_path = os.getenv('LANCEDB_PATH', 'data/lancedb')
        table_name = os.getenv('LANCEDB_TABLE', 'documents')
        
        if not os.path.exists(lancedb_path):
            return jsonify({'error': f'LanceDB path does not exist: {lancedb_path}'}), 404
        
        # Reconnect to LanceDB and reload table
        db = lancedb.connect(lancedb_path)
        if table_name in db.table_names():
            table = db.open_table(table_name)
            return jsonify({
                'status': 'success',
                'message': f'Table {table_name} reloaded successfully',
                'record_count': len(table)
            })
        else:
            return jsonify({'error': f'Table {table_name} not found in {lancedb_path}'}), 404
            
    except Exception as e:
        return jsonify({'error': f'Failed to reload table: {str(e)}'}), 500

if __name__ == '__main__':
    # Initialize retriever in async context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    success = loop.run_until_complete(init_retriever())
    if not success:
        sys.exit(1)
    
    print("Starting retriever HTTP service on port 8002...")
    app.run(host='0.0.0.0', port=8002, debug=False)