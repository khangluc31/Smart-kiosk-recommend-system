# Smart-kiosk-recommend-system

## Introduction
The original idea for this system was developed for the **UEH Business Challenge: AI in Retail**. Although our team did not advance further in the competition to deploy a working model, I decided to continue developing the project independently to bring the core concept to life. This repository contains the ongoing implementation of that intelligent recommendation and workflow optimization system.

## How It Works: The Kiosk Flow

The system simulates the end-to-end customer journey at a self-service fast-food kiosk, heavily leveraging AI and machine learning for personalization and cart optimization.

**Step 1: Wait for a Free Kiosk**
*   Customers wait in line for the next available smart kiosk.

**Step 2: Item Selection & AI Recommendations (Core Function)**
*   **1. Dual Interaction Paths:** Upon approach, the screen offers two flexible experiences: a Traditional flow (touchscreen selection) or an AI Voice Assistant flow. Customers can opt-out at any time by choosing "No AI" or "Do not log in" to order anonymously.
*   **2. AI Voice Assistant:** The Kiosk greets the user and asks for their needs (e.g., budget, cravings) using standard Speech-to-Text (STT) and Text-to-Speech (TTS) APIs. A Transformer model (Encoder-only, like BERT) extracts key intents (budget constraints, preferences, allergies) and converts them into structured data. *(Privacy note: Voice recordings are temporarily cached solely for text extraction and are deleted immediately).*
*   **3. Dynamic Recommendation Routing:** The system checks the customer's account status to route them to the appropriate algorithm:
    *   **Guests/New Customers:** With little to no historical data, the system applies a **LinUCB** (Contextual Bandit) model to suggest top-k meals based on current top sellers and seasonal trends.
    *   **Regular Customers:** The system uses a deep matching model (**DSSM + Faiss**) to retrieve top-k meals matching specific historical preferences.
    *   *Refinement:* Both branches feed into a **LightGBM** model to match the customer's current context with the generated candidates, ensuring highly personalized final recommendations.
*   **4. Choice Presentation:** The Kiosk displays a shortlisted selection (3 to 5 optimal meals) tagged with visual labels like *"Fits your budget"* or *"Familiar choice"*.
*   **5. Cart Optimization:** An **Integer Linear Programming (ILP)** algorithm reviews the selected items and automatically groups single items into cost-saving combos if applicable.
*   **6. Smart Upselling:** 
    *   The system uses **Item2vec + Faiss** to learn item relationships from historical carts. 
    *   It generates a *feasible choice set* based on what is currently in the cart, incorporating best-sellers and Item2Vec suggestions, while strictly filtering out allergies, disliked items, or duplicates.
    *   Before final payment, a second layer of **ILP** evaluates the cart against this feasible set to execute one of two simultaneous strategies:
        *   *Subtle Upsell:* Finding the nearest cost-saving combo (e.g., "Add fries for just 5,000 VND to make it a combo").
        *   *Popular Upsell:* Recommending a highly compatible standalone item.

**Step 3: Order Preparation & Handover**
*   The finalized order is routed directly to the kitchen display system for preparation.
*   Customers wait for their number to be called to receive their order. 

## Important Notes & Limitations

*   **Menu Dependency (October 2026):** Because we do not have access to a live, proprietary database, this project utilizes a static pull of the KFC Vietnam menu from October 2026. The system's mapping may break or behave unpredictably if the menu data structure changes in the future.
*   **Simulated Data:** The system currently relies on a list of dummy users for simulation. Because we lack access to real-world operational data (actual best sellers, hot sellers, real customer histories, and exact order volumes), the current recommendation accuracy will not perfectly reflect real-life consumer behavior. However, the architecture is designed so that if fed with high-quality, real-world data, the models will perform highly effectively.
*   **Language & UI Constraints (Vietnamese Only):** In alignment with the original competition's target demographic, the menu items, intent recognition, and User Interface are entirely in Vietnamese. 
    *   Inputting English text/voice may result in failures or incorrect parsing. 
    *   There is currently no English version developed for the UI or the underlying NLP models.
