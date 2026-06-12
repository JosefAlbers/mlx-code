from __future__ import annotations
import json
from typing import Optional
from pydantic import BaseModel, Field
from .tools import Tool, tout
from .mcb import DocThread

class ReadKBParams(BaseModel):
    id: str = Field(description='Any comment ID to read.')

class ReadKBTool(Tool):
    name = 'ReadKB'
    description = 'Read a specific comment sub-tree from the knowledge base.\n\nPass any comment ID to isolate and view that specific sub-branch and its replies. If the ID is the root of a thread, the branch will contain the document content and its full revision history (diffs).\n\nOutput format when ID is a downstream comment:\n<branch id="comment_id">\n  Comment text and nested replies...\n</branch>\n\nOutput format when ID is the root comment:\n<branch id="root_id">\n  <document rev="vN">Current document content</document>\n  <diff rev="vN">Unified diff and edit history stats</diff>\n  <branch id="child_id">Nested replies...</branch>\n</branch>'
    parameters = ReadKBParams

    async def execute(self, params: ReadKBParams, signal=None) -> dict:
        dt = self.ctx.get('dt')
        if dt is None:
            return tout('No Knowledge base is available', True)
        if params.id not in dt.kb:
            return tout(f'Comment not found: {params.id}', True)
        return tout(dt.read(params.id))

class CommentKBParams(BaseModel):
    content: str = Field(description='Comment content.')
    parent: str = Field(description='Comment ID to reply to.')

class CommentKBTool(Tool):
    name = 'CommentKB'
    description = 'Add a comment reply. Returns the new comment ID.'
    parameters = CommentKBParams

    async def execute(self, params: CommentKBParams, signal=None) -> dict:
        dt = self.ctx.get('dt')
        if dt is None:
            return tout('No knowledge base is available', True)
        if params.parent not in dt.kb:
            return tout(f'Comment not found: {params.parent}', True)
        return tout(json.dumps({'id': dt.comment(params.content, to=params.parent)}))

class SubmitKBParams(BaseModel):
    content: str = Field(description='Full document content.')
    parent: Optional[str] = Field(None, description="Comment ID issuing revision of the thread's document. Omit to submit a new document.")

class SubmitKBTool(Tool):
    name = 'SubmitKB'
    description = "Submit a new synthesis document or update an existing thread.\n\nCRITICAL USAGE:\n- To start a BRAND NEW thread, OMIT the 'parent' parameter entirely. DO NOT pass the string 'None', 'null', or empty string ''.\n- To update an existing document, 'parent' MUST be a valid, pre-existing comment ID."
    parameters = SubmitKBParams

    async def execute(self, params: SubmitKBParams, signal=None) -> dict:
        dt = self.ctx.get('dt')
        if dt is None:
            return tout('No knowledge base is available', True)
        if params.parent is not None and params.parent not in dt.kb:
            return tout(f'comment not found: {params.parent}', True)
        return tout(json.dumps({'id': dt.submit(params.content, parent=params.parent)}))

def system_prompt(dt):
    return f"""You are a Senior Research Architect. When given a goal, you are responsible for the entire lifecycle of the synthesis—from initial survey to final peer-reviewed "Commit."\n\n### **Available Knowledge Landscape (Root IDs)**\n{'\n'.join([f'- {root_id}' for root_id in dt.submissions])}\n(Use ReadKB on these IDs to begin your survey.)\n\n### **1. The Autonomous Project Lifecycle**\n\nUpon receiving a goal, execute these stages in one or more turns:\n\n* **Stage A: Survey & Extraction:** Call `ReadKB` on all relevant Source IDs immediately. Identify core technical claims and architectural shifts.\n* **Stage B: Multi-Perspective Drafting:** Call `SubmitKB` (no parent) to create your baseline synthesis. **Capture the returned ID (the "Thread Root ID").**\n* **Stage C: Delegation & Review (The Handover):** * Spawn a "Reviewer" sub-agent via the `Agent` tool.\n    * **CRITICAL:** You must explicitly pass the **Thread Root ID** (the ID returned by `SubmitKB` in Stage B) to the sub-agent in its task description.\n    * Instruct the sub-agent to use `ReadKB` on that ID and post its critique using `CommentKB`.\n* **Stage D: Adjudication & Rebuttal:** Read the comments on your thread. Reply to the critiques. Defend the draft where the data is solid; plan a correction where the critique is valid.\n* **Stage E: The Final Commit:** Call `SubmitKB` (with the synthesis Root ID as `parent`) to update the document **in-place**.\n\n### **2. Delegation & Sub-Agent Handover Rules**\n\nWhen delegating via the `Agent` tool, your task description must act as a "Project Brief." You are strictly required to:\n1.  **Identify the Workspace:** Provide the specific **Thread Root ID** (the token returned by `SubmitKB`) that the sub-agent must work on.\n2.  **Define the Tool-Chain:** Explicitly tell the agent to use `ReadKB` to see the document and `CommentKB` to post feedback.\n3.  **Set the Objective:** State clearly if the agent is a "Technical Critic," a "Fact Checker," or a "Copy Editor."\n\n### **3. Logic of State (The "Commit" Requirement)**\n\nYou are responsible for the integrity of the "Product" (the Root).\n* **Discussion is Workspace:** Use `CommentKB` for thoughts, extraction notes, and rebuttals.\n* **Submission is Reality:** You haven't "done" anything until it is reflected in the Root node via `SubmitKB`. \n* **The Chain of Custody:** Once you create a synthesis thread, **always** reuse that Thread Root ID (the `SubmitKB` return value) as the `parent` for updates.\n\n### **4. Final Success Criteria**\n\nYou are "Done" only when:\n1. The synthesis Root ID contains the full, finalized text.\n2. The `<diff>` history shows the evolution from your first draft to the final adjudicated version.\n3. All major critiques in the branches have been either **addressed** (via Commit) or **defended** (via Rebuttal).\n"""
ALL_TOOLS = [ReadKBTool, CommentKBTool, SubmitKBTool]
ALL_NAMES = [i.name for i in ALL_TOOLS] + ['Agent']