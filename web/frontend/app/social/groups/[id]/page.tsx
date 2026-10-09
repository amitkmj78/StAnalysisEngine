"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import {
  ApiError, createGroupSession, createPost, getCurrentUser, getGroupSessions, getSocialGroup, getSocialProfile,
  joinSocialGroup, leaveSocialGroup, getPosts, removeGroupMember, removeGroupPost,
} from "@/lib/api";
import type { GroupSession, Post, SocialGroupDetail } from "@/lib/types";

export default function SocialGroupDetailPage() {
  const params = useParams<{ id: string }>();
  const groupId = Number(params.id);

  const [group, setGroup] = useState<SocialGroupDetail | null>(null);
  const [posts, setPosts] = useState<Post[] | null>(null);
  const [myUserId, setMyUserId] = useState<string | null>(null);
  const [isMentor, setIsMentor] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [body, setBody] = useState("");

  // BEG-5: group sessions.
  const [sessions, setSessions] = useState<GroupSession[] | null>(null);
  const [sessionTitle, setSessionTitle] = useState("");
  const [sessionDescription, setSessionDescription] = useState("");
  const [sessionWhen, setSessionWhen] = useState("");
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [creatingSession, setCreatingSession] = useState(false);

  async function load() {
    try {
      const [g, p, me, s] = await Promise.all([
        getSocialGroup(groupId), getPosts({ group_id: groupId }), getCurrentUser(), getGroupSessions(groupId),
      ]);
      setGroup(g);
      setPosts(p.posts);
      setMyUserId(me.id);
      setSessions(s.sessions);
      const myProfile = await getSocialProfile(me.id);
      setIsMentor(myProfile.mentor_badge);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Failed to load this group.");
    }
  }

  async function handleCreateSession() {
    if (!sessionTitle.trim() || !sessionWhen) return;
    setCreatingSession(true);
    setSessionError(null);
    try {
      await createGroupSession(groupId, {
        title: sessionTitle.trim(),
        description: sessionDescription.trim() || undefined,
        scheduled_at: new Date(sessionWhen).toISOString(),
      });
      setSessionTitle("");
      setSessionDescription("");
      setSessionWhen("");
      const s = await getGroupSessions(groupId);
      setSessions(s.sessions);
    } catch (err) {
      setSessionError(err instanceof ApiError ? err.message : "Could not create this session.");
    } finally {
      setCreatingSession(false);
    }
  }

  useEffect(() => {
    load();
  }, [groupId]);

  const myRole = group?.members.find((m) => m.user_id === myUserId)?.role;
  const isModerator = myRole === "moderator" || myRole === "owner";
  const isMember = !!myRole;

  async function handlePost() {
    const text = body.trim();
    if (!text) return;
    try {
      await createPost({ body: text, group_id: groupId });
      setBody("");
      await load();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not post.");
    }
  }

  async function handleJoin() {
    await joinSocialGroup(groupId);
    await load();
  }

  async function handleLeave() {
    await leaveSocialGroup(groupId);
    await load();
  }

  async function handleRemoveMember(memberId: string) {
    await removeGroupMember(groupId, memberId);
    await load();
  }

  async function handleRemovePost(postId: number) {
    await removeGroupPost(groupId, postId);
    await load();
  }

  if (error) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-red-700">{error}</div>;
  if (!group) return <div className="mx-auto max-w-3xl px-4 py-8 text-sm text-slate-500">Loading…</div>;

  return (
    <div className="mx-auto max-w-3xl px-4 py-8">
      <div className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-semibold text-slate-900">
          {group.name}
          {group.is_private && <span className="ml-2 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600">PRIVATE</span>}
        </h1>
        <Link href="/social/groups" className="text-sm font-medium text-slate-600 hover:underline">All groups</Link>
      </div>
      {group.description && <p className="mt-1 text-sm text-slate-600">{group.description}</p>}
      <div className="mt-2 flex items-center gap-3 text-sm">
        {!isMember && !group.is_private && (
          <button onClick={handleJoin} className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-50">Join</button>
        )}
        {isMember && myRole !== "owner" && (
          <button onClick={handleLeave} className="rounded-md border border-slate-300 px-3 py-1 text-xs hover:bg-slate-50">Leave</button>
        )}
      </div>

      <div className="mt-4">
        <h2 className="text-sm font-semibold text-slate-900">Members ({group.members.length})</h2>
        <div className="mt-1 flex flex-col gap-1">
          {group.members.map((m) => (
            <div key={m.user_id} className="flex items-center justify-between text-sm">
              <span>
                {m.display_name ?? "User"} <span className="text-xs text-slate-400">({m.role})</span>
              </span>
              {isModerator && m.role !== "owner" && m.user_id !== myUserId && (
                <button onClick={() => handleRemoveMember(m.user_id)} className="text-xs text-red-600 hover:underline">Remove</button>
              )}
            </div>
          ))}
        </div>
      </div>

      <div className="mt-4">
        <h2 className="text-sm font-semibold text-slate-900">Sessions</h2>
        {sessions === null && <p className="mt-1 text-sm text-slate-500">Loading…</p>}
        {sessions && sessions.length === 0 && (
          <p className="mt-1 text-sm text-slate-500">No sessions scheduled yet.</p>
        )}
        {sessions && sessions.length > 0 && (
          <div className="mt-1 flex flex-col gap-2">
            {sessions.map((s) => (
              <div key={s.id} className="rounded-md border border-slate-100 p-3 text-sm">
                <div className="flex items-center justify-between">
                  <strong className="text-slate-800">{s.title}</strong>
                  <span className="text-xs text-slate-400">{new Date(s.scheduled_at).toLocaleString()}</span>
                </div>
                {s.description && <p className="mt-1 text-slate-600">{s.description}</p>}
                <p className="mt-1 text-xs text-slate-400">Hosted by {s.host_display_name ?? "a mentor"}</p>
              </div>
            ))}
          </div>
        )}

        {isMentor && isMember && (
          <div className="mt-3 rounded-md border border-slate-200 bg-slate-50 p-3">
            <p className="text-xs font-medium text-slate-600">Host a session (mentors only)</p>
            <div className="mt-2 flex flex-col gap-2">
              <input
                value={sessionTitle}
                onChange={(e) => setSessionTitle(e.target.value)}
                placeholder="Session title"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                value={sessionDescription}
                onChange={(e) => setSessionDescription(e.target.value)}
                placeholder="Description (optional)"
                className="rounded-md border border-slate-300 px-2 py-1 text-sm"
              />
              <input
                type="datetime-local"
                value={sessionWhen}
                onChange={(e) => setSessionWhen(e.target.value)}
                className="rounded-md border border-slate-300 px-2 py-1 text-sm"
              />
              <button
                onClick={handleCreateSession}
                disabled={creatingSession || !sessionTitle.trim() || !sessionWhen}
                className="self-start rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50"
              >
                {creatingSession ? "Scheduling…" : "Schedule session"}
              </button>
              {sessionError && <p className="text-xs text-red-700">{sessionError}</p>}
            </div>
          </div>
        )}
      </div>

      {isMember && (
        <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
          <textarea value={body} onChange={(e) => setBody(e.target.value)} placeholder={`Post in ${group.name}...`} className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm" rows={2} />
          <button onClick={handlePost} disabled={!body.trim()} className="mt-2 rounded-md bg-slate-900 px-3 py-1.5 text-sm font-medium text-white disabled:opacity-50">Post</button>
        </div>
      )}

      <div className="mt-6 flex flex-col gap-3">
        {posts?.map((p) => (
          <div key={p.id} className="rounded-xl border border-slate-200 bg-white p-4 text-sm">
            <div className="flex items-center justify-between">
              <strong className="text-slate-800">{p.display_name}</strong>
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">{new Date(p.created_at).toLocaleString()}</span>
                {isModerator && (
                  <button onClick={() => handleRemovePost(p.id)} className="text-xs text-red-600 hover:underline">Remove</button>
                )}
              </div>
            </div>
            <p className="mt-1 text-slate-700">{p.body}</p>
          </div>
        ))}
        {posts?.length === 0 && <p className="text-sm text-slate-500">No posts in this group yet.</p>}
      </div>
    </div>
  );
}
