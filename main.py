import asyncio
import websockets
import json
import os
import aiohttp
import hashlib
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VideoStreamServer:
    def __init__(self, host='localhost', port=1738, upload_dir='uploads'):
        self.host = host
        self.port = port
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.active_uploads = {}
        self.clients = set()
        
    async def handle_client(self, websocket):
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New client connected: {client_id}")
        self.clients.add(websocket)
        
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self.handle_upload_chunk_binary(websocket, message, client_id)
                else:
                    await self.process_message(websocket, message, client_id)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_id}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {str(e)}")
            await self.send_error(websocket, str(e))
        finally:
            self.clients.discard(websocket)
            if client_id in self.active_uploads:
                await self.cleanup_upload(client_id)
    
    async def process_message(self, websocket, message, client_id):
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'start_upload':
                await self.handle_start_upload(websocket, data, client_id)
            elif msg_type == 'cancel_upload':
                await self.handle_cancel_upload(websocket, data, client_id)
            elif msg_type == 'ping':
                await self.send_response(websocket, {'type': 'pong'})
            else:
                await self.send_error(websocket, f"Unknown message type: {msg_type}")
                
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            await self.send_error(websocket, str(e))
    
    async def handle_start_upload(self, websocket, data, client_id):
        try:
            filename = data.get('filename', 'video.mp4')
            filesize = data.get('filesize', 0)
            checksum = data.get('checksum', '')
            total_chunks = data.get('totalChunks', 0)
            
            if not filename or filesize <= 0:
                await self.send_error(websocket, "Invalid file information")
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_filename = f"{timestamp}_{filename}".replace('..', '').replace('/', '_')
            temp_path = self.upload_dir / f".{safe_filename}.tmp"
            final_path = self.upload_dir / safe_filename
            
            self.active_uploads[client_id] = {
                'filename': filename,
                'safe_filename': safe_filename,
                'filesize': filesize,
                'checksum': checksum,
                'temp_path': temp_path,
                'final_path': final_path,
                'temp_file': open(temp_path, 'wb'),
                'bytes_received': 0,
                'chunks_received': 0,
                'total_chunks': total_chunks,
                'start_time': datetime.now(),
                'hasher': hashlib.sha256() if checksum else None
            }
            
            logger.info(f"Starting upload: {filename} ({filesize} bytes, {total_chunks} chunks) from {client_id}")
            
            await self.send_response(websocket, {
                'type': 'upload_started',
                'upload_id': client_id,
                'message': 'Upload initialized successfully'
            })
            
        except Exception as e:
            logger.error(f"Error starting upload: {str(e)}")
            await self.send_error(websocket, f"Failed to start upload: {str(e)}")
    
    async def handle_upload_chunk_binary(self, websocket, chunk_data, client_id):
        upload_info = self.active_uploads.get(client_id)
        if not upload_info:
            logger.warning("Late chunk from %s ignored", client_id)
            return

        try:
            upload_info['temp_file'].write(chunk_data)
            upload_info['bytes_received'] += len(chunk_data)
            upload_info['chunks_received'] += 1
            
            if upload_info['hasher']:
                upload_info['hasher'].update(chunk_data)

            progress = (upload_info['bytes_received'] / upload_info['filesize']) * 100
            
            logger.debug(f"Received chunk {upload_info['chunks_received']}/{upload_info['total_chunks']} "
                        f"({progress:.1f}%) for {upload_info['filename']}")

            if (upload_info['bytes_received'] >= upload_info['filesize'] and upload_info['chunks_received'] >= upload_info['total_chunks']):
                await self.complete_upload(websocket, client_id)
            else:
                await self.send_response(websocket, {
                    'type': 'chunk_received',
                    'chunk_number': upload_info['chunks_received'],
                    'bytes_received': upload_info['bytes_received'],
                    'progress': progress
                })
                
        except IOError as e:
            logger.error(f"Failed to write chunk: {str(e)}")
            await self.send_error(websocket, "Failed to write chunk to disk")
        except Exception as e:
            logger.error(f"Error handling chunk: {str(e)}")
            await self.send_error(websocket, f"Failed to process chunk: {str(e)}")
    
    async def complete_upload(self, websocket, client_id):
        upload_info = self.active_uploads[client_id]
        
        try:
            upload_info['temp_file'].close()
            
            actual_size = os.path.getsize(upload_info['temp_path'])
            if actual_size != upload_info['filesize']:
                os.unlink(upload_info['temp_path'])
                await self.send_error(websocket, 
                    f"Size mismatch - expected {upload_info['filesize']}, got {actual_size}")
                return
            
            if upload_info['checksum'] and upload_info['hasher']:
                calculated_checksum = upload_info['hasher'].hexdigest()
                if calculated_checksum != upload_info['checksum']:
                    os.unlink(upload_info['temp_path'])
                    await self.send_error(websocket, "Checksum mismatch - file corrupted")
                    return
            
            os.rename(upload_info['temp_path'], upload_info['final_path'])
                        
            duration = (datetime.now() - upload_info['start_time']).total_seconds()
            
            logger.info(f"Upload completed: {upload_info['filename']} "
                       f"({actual_size} bytes in {duration:.1f}s)")
            
            await self.send_response(websocket, {
                'type': 'upload_completed',
                'filename': upload_info['safe_filename'],
                'filesize': actual_size,
                'duration': duration,
                'message': 'Video uploaded successfully'
            })
            
            asyncio.create_task(self.run_detection(upload_info["safe_filename"]))

            upload_info['finished'] = True
            await asyncio.sleep(5)
            self.active_uploads.pop(client_id, None)
                        
        except Exception as e:
            logger.error(f"Error completing upload: {str(e)}")
            await self.send_error(websocket, f"Failed to complete upload: {str(e)}")
            await self.cleanup_upload(client_id)
    
    async def handle_cancel_upload(self, websocket, data, client_id):
        if client_id not in self.active_uploads:
            await self.send_response(websocket, {
                'type': 'upload_cancelled',
                'message': 'No active upload to cancel'
            })
            return
        
        try:
            upload_info = self.active_uploads[client_id]
            logger.info(f"Upload cancelled: {upload_info['filename']} from {client_id}")
            
            await self.cleanup_upload(client_id)
            
            await self.send_response(websocket, {
                'type': 'upload_cancelled',
                'message': 'Upload cancelled successfully'
            })
            
        except Exception as e:
            logger.error(f"Error cancelling upload: {str(e)}")
            await self.send_error(websocket, f"Failed to cancel upload: {str(e)}")
    
    async def cleanup_upload(self, client_id):
        if client_id in self.active_uploads:
            upload_info = self.active_uploads[client_id]
            try:
                if upload_info.get('temp_file') and not upload_info['temp_file'].closed:
                    upload_info['temp_file'].close()
                if upload_info.get('temp_path') and os.path.exists(upload_info['temp_path']):
                    os.unlink(upload_info['temp_path'])
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
            finally:
                del self.active_uploads[client_id]
    
    async def send_response(self, websocket, data):
        try:
            await websocket.send(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending response: {str(e)}")
    
    async def send_error(self, websocket, error_message):
        await self.send_response(websocket, {
            'type': 'error',
            'message': error_message
        })
    
    async def start(self):
        logger.info(f"Starting video streaming server on {self.host}:{self.port}")
        logger.info(f"Upload directory: {self.upload_dir.absolute()}")
        
        async with websockets.serve(
            self.handle_client, 
            self.host, 
            self.port,
            max_size=10 * 1024 * 1024,
            compression=None,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=10
        ):
            await asyncio.Future()

    async def run_detection(self, filename):
        try:
            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:8000/analyze_video?filename={filename}"
                async with session.post(url) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Gaze detection run on uploaded file: {filename}, see output file for results")
                        with open('results.json', 'w') as output_file:
                            json.dump(result, output_file, indent=2)
                    else:
                        logger.error(f"API call failed, status code: {response.status}, filename: {filename}")
                        error_text = await response.text()
                        logger.error(f"Error text: {error_text}")
        except aiohttp.ClientError as e:
            logger.error(f"Network error calling gaze results api for file {filename}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error calling gaze results api for file {filename}: {str(e)}")
            
def main():
    server = VideoStreamServer(
        host='127.0.0.1',
        port=1738,
        upload_dir='uploads'
    )
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")

if __name__ == '__main__':
    main()