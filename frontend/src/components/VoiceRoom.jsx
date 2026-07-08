import React, { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  LiveKitRoom,
  RoomAudioRenderer,
  ControlBar,
  useVoiceAssistant,
  useConnectionState,
  useRoomContext,
  useLocalParticipant,
  useParticipants,
} from '@livekit/components-react';
import { getToken } from '../api';
import './VoiceRoom.css';

const VoiceRoom = () => {
  const [roomName, setRoomName] = useState('test-room');
  const [identity, setIdentity] = useState(`user-${Math.random().toString(36).substring(2, 8)}`);
  const [isConnected, setIsConnected] = useState(false);
  const [token, setToken] = useState(null);
  const [url, setUrl] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [selectedDoc, setSelectedDoc] = useState(null);

  // Load selected document from localStorage
  useEffect(() => {
    const docId = localStorage.getItem('selectedDocumentId');
    const docName = localStorage.getItem('selectedDocumentName');
    if (docId && docName) {
      setSelectedDoc({ id: docId, name: docName });
    }
  }, []);

  const handleConnect = async () => {
    try {
      console.log("[UI] Connect button clicked");
      console.log("[UI] Room name:", roomName);
      console.log("[UI] Identity:", identity);
      console.log("[UI] Selected Doc:", selectedDoc);
      
      setStatus('Connecting...');
      
      // Prepare metadata
      const metadata = selectedDoc ? {
        user_id: identity,
        document_id: selectedDoc.id,
        document_name: selectedDoc.name
      } : {
        user_id: identity,
        document_id: 'default_doc'
      };
      
      const data = await getToken(identity, roomName, metadata);
      
      console.log("[TOKEN RESPONSE]", data);
      console.log("[LIVEKIT URL]", data.url);
      
      setToken(data.token);
      setUrl(data.url);
      setIsConnected(true);
      setStatus('Connected');
    } catch (error) {
      console.error('[ERROR] Failed to connect:', error);
      setStatus('Connection failed');
    }
  };

  return (
    <div className="voice-room">
      <header className="voice-room-header fade-in">
        <div className="logo">
          <div className="logo-icon">🎙️</div>
          <div className="logo-text">
            <h1>AI Voice Agent</h1>
            <p>Real-time voice assistant</p>
          </div>
        </div>
        <div className="header-right">
          {selectedDoc && (
            <div className="selected-doc-badge">
              <span className="doc-icon">📄</span>
              <span className="doc-name">{selectedDoc.name}</span>
            </div>
          )}
          <Link to="/documents" className="manage-docs-button">
            Manage Documents
          </Link>
          <div className="connection-status">
            <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}></div>
            <span>{status}</span>
          </div>
        </div>
      </header>

      <main className="voice-room-main">
        {!isConnected ? (
          <div className="join-form fade-in">
            <div className="form-card glass">
              <h2>Join a Room</h2>
              {selectedDoc ? (
                <div className="doc-info">
                  <div className="doc-info-icon">📄</div>
                  <div className="doc-info-text">
                    <strong>Active Document:</strong>
                    <div>{selectedDoc.name}</div>
                  </div>
                </div>
              ) : (
                <div className="doc-warning">
                  ⚠️ No document selected. The AI will use default document.
                </div>
              )}
              <div className="form-group">
                <label htmlFor="roomName">Room Name</label>
                <input
                  type="text"
                  id="roomName"
                  value={roomName}
                  onChange={(e) => setRoomName(e.target.value)}
                  placeholder="Enter room name"
                  className="input-field"
                />
              </div>
              <div className="form-group">
                <label htmlFor="identity">Your Identity</label>
                <input
                  type="text"
                  id="identity"
                  value={identity}
                  onChange={(e) => setIdentity(e.target.value)}
                  placeholder="Enter your identity"
                  className="input-field"
                />
              </div>
              <button onClick={handleConnect} className="join-button">
                Join Room
              </button>
            </div>
          </div>
        ) : (
          <LiveKitRoom
            token={token}
            serverUrl={url}
            connect={true}
            video={false}
            audio={true}
            style={{ width: '100%', height: '100%' }}
          >
            <ConnectedView setStatus={setStatus} roomNameFromParent={roomName} />
            <RoomAudioRenderer />
            <div className="control-bar-wrapper">
              <ControlBar />
            </div>
          </LiveKitRoom>
        )}
      </main>
    </div>
  );
};

// Connected View Component
const ConnectedView = ({ setStatus, roomNameFromParent }) => {
  const connectionState = useConnectionState();
  const voiceAssistant = useVoiceAssistant();
  
  // Log ALL keys from voiceAssistant to find correct state property
  useEffect(() => {
    console.log("[VOICE ASSISTANT FULL OBJECT]", voiceAssistant);
    if (voiceAssistant) {
      console.log("[VOICE ASSISTANT KEYS]", Object.keys(voiceAssistant));
    }
  }, [voiceAssistant]);

  // Try all possible state properties
  let state = 'idle';
  if (voiceAssistant) {
    if (voiceAssistant.state) state = voiceAssistant.state;
    if (voiceAssistant.agent?.state) state = voiceAssistant.agent.state;
    if (voiceAssistant.agentState) state = voiceAssistant.agentState;
    if (voiceAssistant.agent_state) state = voiceAssistant.agent_state;
  }

  const messages = voiceAssistant?.messages || [];
  const transcriptEndRef = useRef(null);
  const roomContext = useRoomContext();
  const { localParticipant, microphoneTrack } = useLocalParticipant();
  const remoteParticipants = useParticipants();

  useEffect(() => {
    console.log("[FINAL VOICE ASSISTANT STATE]", state);
  }, [state]);

  // Log room info and LIVEKIT URL match with worker
  useEffect(() => {
    if (roomContext) {
      console.log("[ROOM CONTEXT]", roomContext);
      console.log("[ROOM NAME FROM LIVEKIT]", roomContext.name);
      console.log("[ROOM NAME FROM UI]", roomNameFromParent);
    }
  }, [roomContext, roomNameFromParent]);

  // Log connection state continuously
  useEffect(() => {
    console.log("[ROOM CONNECTION STATE]", connectionState);
    setStatus(connectionState === 'connected' ? 'Connected' : connectionState);
  }, [connectionState, setStatus]);

  // Log room and local participant/track info with diagnostics
  useEffect(() => {
    if (localParticipant) {
      console.log("[LOCAL PARTICIPANT]", localParticipant);
      if (microphoneTrack) {
        console.log("[MIC TRACK] Published Successfully", {
          isMuted: microphoneTrack.isMuted,
          name: microphoneTrack.trackName,
          kind: microphoneTrack.kind
        });
      }
    }
  }, [localParticipant, microphoneTrack]);

  // Log remote participants (should include the agent!)
  useEffect(() => {
    console.log("[REMOTE PARTICIPANTS COUNT]", remoteParticipants.length);
    if (remoteParticipants.length > 0) {
      remoteParticipants.forEach((rp) => {
        console.log("[REMOTE PARTICIPANT]", rp.identity, rp);
      });
    }
  }, [remoteParticipants]);

  // Debugging logs for room events
  useEffect(() => {
    if (!roomContext) return;

    const handleParticipantConnected = (participant) => {
      console.log("[DEBUG] Participant joined:", participant.identity);
    };

    const handleParticipantDisconnected = (participant) => {
      console.log("[DEBUG] Participant left:", participant.identity);
    };

    const handleTrackSubscribed = (track, publication, participant) => {
      console.log("[DEBUG] Track subscribed:", participant.identity, track.kind);
    };

    const handleTrackUnsubscribed = (track, publication, participant) => {
      console.log("[DEBUG] Track unsubscribed:", participant.identity, track.kind);
    };

    roomContext.on('participantConnected', handleParticipantConnected);
    roomContext.on('participantDisconnected', handleParticipantDisconnected);
    roomContext.on('trackSubscribed', handleTrackSubscribed);
    roomContext.on('trackUnsubscribed', handleTrackUnsubscribed);

    return () => {
      roomContext.off('participantConnected', handleParticipantConnected);
      roomContext.off('participantDisconnected', handleParticipantDisconnected);
      roomContext.off('trackSubscribed', handleTrackSubscribed);
      roomContext.off('trackUnsubscribed', handleTrackUnsubscribed);
    };
  }, [roomContext]);

  // Auto scroll to bottom of transcripts
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="connected-view">
      {/* Remote Participants Section */}
      <div className="participants-section">
        <h3>Participants</h3>
        <div className="participants-grid">
          {remoteParticipants.map((participant) => (
            <ParticipantView key={participant.identity} participant={participant} />
          ))}
        </div>
      </div>

      <div className="visualization-section">
        <VoiceVisualization state={state} />
        <div className="state-indicator">
          <div className="state-icon">{getStateIcon(state)}</div>
          <div className="state-text">{getStateText(state)}</div>
        </div>
      </div>

      <div className="transcripts-panel glass">
        <h3>Conversation</h3>
        <div className="transcripts-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">💬</div>
              <p>Start speaking to see your conversation here</p>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <Message key={idx} message={msg} />
            ))
          )}
          <div ref={transcriptEndRef} />
        </div>
      </div>
    </div>
  );
};

// Message Component
const Message = ({ message }) => {
  if (!message || !message.text) return null;
  
  const isUser = message.role === 'user' || message.type === 'user';
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className={`message ${isUser ? 'user-message' : 'agent-message'} slide-in`}>
      <div className="message-avatar">
        {isUser ? '👤' : '🤖'}
      </div>
      <div className="message-content">
        <div className="message-header">
          <span className="message-author">{isUser ? 'You' : 'Assistant'}</span>
          <span className="message-time">{time}</span>
        </div>
        <div className="message-text">{message.text}</div>
      </div>
    </div>
  );
};

// Voice Visualization Component
const VoiceVisualization = ({ state }) => {
  const isListening = state === 'listening';
  const isSpeaking = state === 'speaking';

  return (
    <div className={`visualization-container ${state === 'idle' ? 'breathe' : ''}`}>
      <div className={`voice-orb ${isListening ? 'listening' : ''} ${isSpeaking ? 'speaking' : ''}`}>
        <div className="pulse-ring"></div>
        <div className="pulse-ring"></div>
        <div className="pulse-ring"></div>
        <div className="mic-icon">{isSpeaking ? '🤖' : '🎤'}</div>
      </div>
      <div className="wave-form">
        {[...Array(12)].map((_, i) => (
          <div
            key={i}
            className={`wave-bar ${isListening || isSpeaking ? 'active' : ''}`}
            style={{
              animation: `wave 0.8s ease-in-out ${i * 0.05}s infinite`
            }}
          />
        ))}
      </div>
    </div>
  );
};

// Helper Functions
const getStateIcon = (state) => {
  switch (state) {
    case 'listening': return '👂';
    case 'speaking': return '🔊';
    case 'thinking': return '🤔';
    default: return '🎧';
  }
};

const getStateText = (state) => {
  switch (state) {
    case 'listening': return 'Listening...';
    case 'speaking': return 'Speaking...';
    case 'thinking': return 'Thinking...';
    default: return 'Ready to talk';
  }
};

// Participant View Component
const ParticipantView = ({ participant }) => {
  const videoRef = useRef(null);
  const audioRef = useRef(null);

  useEffect(() => {
    const handleTrackPublished = (publication) => {
      console.log("[ParticipantView] Track published:", publication.trackSid, publication.kind);
    };

    const handleTrackSubscribed = (track) => {
      console.log("[ParticipantView] Track subscribed:", track.sid, track.kind);
      if (track.kind === 'video' && videoRef.current) {
        track.attach(videoRef.current);
      }
      if (track.kind === 'audio' && audioRef.current) {
        track.attach(audioRef.current);
      }
    };

    const handleTrackUnsubscribed = (track) => {
      console.log("[ParticipantView] Track unsubscribed:", track.sid, track.kind);
      if (track.kind === 'video' && videoRef.current) {
        track.detach(videoRef.current);
      }
      if (track.kind === 'audio' && audioRef.current) {
        track.detach(audioRef.current);
      }
    };

    participant.on('trackPublished', handleTrackPublished);
    participant.on('trackSubscribed', handleTrackSubscribed);
    participant.on('trackUnsubscribed', handleTrackUnsubscribed);

    // Attach already subscribed tracks
    if (participant.tracks) {
      const subscribedTracks = Array.from(participant.tracks.values());
      subscribedTracks.forEach((publication) => {
        if (publication.track) {
          handleTrackSubscribed(publication.track);
        }
      });
    }

    return () => {
      participant.off('trackPublished', handleTrackPublished);
      participant.off('trackSubscribed', handleTrackSubscribed);
      participant.off('trackUnsubscribed', handleTrackUnsubscribed);
    };
  }, [participant]);

  return (
    <div className="participant-tile-wrapper">
      <div className="participant-tile">
        <video ref={videoRef} className="participant-video" autoPlay playsInline muted />
        <audio ref={audioRef} autoPlay />
      </div>
      <div className="participant-name">{participant.identity}</div>
    </div>
  );
};

export default VoiceRoom;
