# reviewer_node.py
import os
from utils import save_file, FoamPydantic
from pydantic import BaseModel, Field
from typing import List

REVIEWER_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to review the provided error logs and diagnose the underlying issues. "
    "You will be provided with a similar case reference, which is a list of similar cases that are ordered by similarity. You can use this reference to help you understand the user requirement and the error."
    "When an error indicates that a specific keyword is undefined (for example, 'div(phi,(p|rho)) is undefined'), your response must propose a solution that simply defines that exact keyword as shown in the error log. "
    "Do not reinterpret or modify the keyword (e.g., do not treat '|' as 'or'); instead, assume it is meant to be taken literally. "
    "Propose ideas on how to resolve the errors, but do not modify any files directly. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead. Do not ask the user any questions."
    "The user will supply all relevant foam files along with the error logs, and within the logs, you will find both the error content and the corresponding error command indicated by the log file name."
)

REWRITE_SYSTEM_PROMPT = (
    "You are an expert in OpenFOAM simulation and numerical modeling. "
    "Your task is to modify and rewrite the necessary OpenFOAM files to fix the reported error. "
    "Please do not propose solutions that require modifying any parameters declared in the user requirement, try other approaches instead."
    "The user will provide the error content, error command, reviewer's suggestions, and all relevant foam files. "
    "Only return files that require rewriting, modification, or addition; do not include files that remain unchanged. "
    "Return the complete, corrected file contents in the following JSON format: "
    "list of foamfile: [{file_name: 'file_name', folder_name: 'folder_name', content: 'content'}]. "
    "Ensure your response includes only the modified file content with no extra text, as it will be parsed using Pydantic."
)



def reviewer_node(state):
    """
    Reviewer node: Reviews the error logs and determines if the error
    is related to the input file. 
    """
    config = state["config"]
    
    print(f"============================== Reviewer Analysis ==============================")
    if len(state["error_logs"]) == 0:
        print("No error to review.")
        return state
    
    # Analysis the reason and give the method to fix the error.
    if state.get("history_text") and state["history_text"]:
        reviewer_user_prompt = (
            f"<similar_case_reference>{state['tutorial_reference']}</similar_case_reference>\n"
            f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
            f"<current_error_logs>{state['error_logs']}</current_error_logs>\n"
            f"<history>\n"
            f"{chr(10).join(state['history_text'])}\n"
            f"</history>\n\n"
            f"<user_requirement>{state['user_requirement']}</user_requirement>\n\n"
            f"I have modified the files according to your previous suggestions. If the error persists, please provide further guidance. Make sure your suggestions adhere to user requirements and do not contradict it. Also, please consider the previous attempts and try a different approach."
        )
    else:
        reviewer_user_prompt = (
            f"<similar_case_reference>{state['tutorial_reference']}</similar_case_reference>\n"
            f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
            f"<error_logs>{state['error_logs']}</error_logs>\n"
            f"<user_requirement>{state['user_requirement']}</user_requirement>\n"
            "Please review the error logs and provide guidance on how to resolve the reported errors. Make sure your suggestions adhere to user requirements and do not contradict it."
        ) 
    
    review_response = state["llm_service"].invoke(reviewer_user_prompt, REVIEWER_SYSTEM_PROMPT)
    review_content = review_response
    
    # Initialize history_text if it doesn't exist
    if not state.get("history_text"):
        history_text = []
    else:
        history_text = state["history_text"]
        
    # Add current attempt to history
    current_attempt = [
        f"<Attempt {len(history_text)//4 + 1}>\n"
        f"<Error_Logs>\n{state['error_logs']}\n</Error_Logs>",
        f"<Review_Analysis>\n{review_content}\n</Review_Analysis>",
        f"</Attempt>\n"  # Closing tag for Attempt with empty line
    ]
    history_text.extend(current_attempt)
    
    
    print(review_content)

    # Return the revised foamfile content.
    rewrite_user_prompt = (
        f"<foamfiles>{str(state['foamfiles'])}</foamfiles>\n"
        f"<error_logs>{state['error_logs']}</error_logs>\n"
        f"<reviewer_analysis>{review_content}</reviewer_analysis>\n\n"
        f"<user_requirement>{state['user_requirement']}</user_requirement>\n\n"
        "Please update the relevant OpenFOAM files to resolve the reported errors, ensuring that all modifications strictly adhere to the specified formats. Ensure all modifications adhere to user requirement."
    )
    rewrite_response = state["llm_service"].invoke(rewrite_user_prompt, REWRITE_SYSTEM_PROMPT, pydantic_obj=FoamPydantic)
    
    # Save the modified files.
    print(f"============================== Rewrite ==============================")
    for foamfile in rewrite_response.list_foamfile:
        print(f"Modified the file: {foamfile.file_name} in folder: {foamfile.folder_name}")
        file_path = os.path.join(state["case_dir"], foamfile.folder_name, foamfile.file_name)
        save_file(file_path, foamfile.content)
        
        # Update state
        if foamfile.folder_name not in state["dir_structure"]:
            state["dir_structure"][foamfile.folder_name] = []
        if foamfile.file_name not in state["dir_structure"][foamfile.folder_name]:
            state["dir_structure"][foamfile.folder_name].append(foamfile.file_name)
        
        for f in state["foamfiles"].list_foamfile:
            if f.folder_name == foamfile.folder_name and f.file_name == foamfile.file_name:
                state["foamfiles"].list_foamfile.remove(f)
                break
            
        state["foamfiles"].list_foamfile.append(foamfile)
    
    # Return updated state
    return {
        **state,
        "history_text": history_text,
        "error_logs": []  # Clear errors after fixing
    }
