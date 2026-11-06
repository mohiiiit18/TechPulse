-- Migration: 001_initial_schema.sql
-- Description: Idempotent migration to initialize/upgrade TechPulse database schema.
-- Safety: Preserves all existing tables and data; applies non-destructive conversions.

-- 1. Ensure users table exists
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- 2. Ensure user_profiles table exists
CREATE TABLE IF NOT EXISTS user_profiles (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    skills TEXT[] DEFAULT '{}',
    target_role VARCHAR(100),
    preferred_location VARCHAR(100),
    experience_level VARCHAR(100),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);

-- Safe migration for user_profiles if skills was previously TEXT
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'user_profiles' AND column_name = 'skills' AND data_type = 'text'
    ) THEN
        -- Convert text to text[] safely
        ALTER TABLE user_profiles 
        ALTER COLUMN skills TYPE TEXT[] 
        USING CASE 
            WHEN skills IS NULL THEN '{}'::text[]
            WHEN skills LIKE '[%' THEN (
                SELECT ARRAY_AGG(x)::text[] 
                FROM json_array_elements_text(skills::json) x
            )
            ELSE string_to_array(skills, ',')
        END;
    END IF;
END $$;

-- 3. Ensure analysis_history table exists
CREATE TABLE IF NOT EXISTS analysis_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_type VARCHAR(50) NOT NULL,
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_analysis_history_user_date ON analysis_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_history_type ON analysis_history(analysis_type);

-- Safe migration for analysis_history if result was previously TEXT
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'analysis_history' AND column_name = 'result' AND data_type = 'text'
    ) THEN
        ALTER TABLE analysis_history 
        ALTER COLUMN result TYPE JSONB 
        USING CASE 
            WHEN result IS NULL THEN '{}'::jsonb
            WHEN result = '' THEN '{}'::jsonb
            ELSE result::jsonb 
        END;
    END IF;
END $$;

-- 4. Ensure saved_jobs table exists
CREATE TABLE IF NOT EXISTS saved_jobs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    job_id VARCHAR(500) NOT NULL,
    saved_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_saved_job UNIQUE(user_id, job_id)
);
CREATE INDEX IF NOT EXISTS idx_saved_jobs_user_date ON saved_jobs(user_id, saved_at DESC);

-- 5. Ensure chat_messages table exists
CREATE TABLE IF NOT EXISTS chat_messages (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    message TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(user_id, session_id, created_at ASC);

-- 6. Ensure feedback table exists
CREATE TABLE IF NOT EXISTS feedback (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    comments TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at DESC);

