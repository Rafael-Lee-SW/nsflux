import { type NextRequest, NextResponse } from "next/server"

// Backend API URL (change to your actual server URL)
const BACKEND_API_URL = process.env.BACKEND_API_URL || "http://localhost:5000"

export async function POST(request: NextRequest) {
    console.log("Stop request received - server route")

    try {
        // Parse JSON request from client
        const requestData = await request.json()

        console.log("Stop request data:", requestData)

        if (!requestData.request_id) {
            console.error("request_id missing")
            return NextResponse.json({ error: "request_id is required" }, { status: 400 })
        }

        console.log(`Generation stop request: ${requestData.request_id}`)

        try {
            // Forward stop request to backend server
            console.log(`Sending stop request to backend: ${BACKEND_API_URL}/stop_generation`)

            const backendResponse = await fetch(`${BACKEND_API_URL}/stop_generation`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestData),
                // Set timeout (5 seconds)
                signal: AbortSignal.timeout(5000),
            })

            // Process response
            console.log(`Backend response status: ${backendResponse.status}`)
            if (!backendResponse.ok) {
                console.error(`Backend server response error: ${backendResponse.status}`)
                return NextResponse.json(
                    {
                        success: true, // Send success to client
                        error: `Backend server error: ${backendResponse.status}`,
                        message: "Stop request was sent but backend returned an error. UI will update to stopped state.",
                    },
                    { status: 200 }, // Respond with 200 to client
                )
            }

            // Success response
            console.log("Received stop success response from backend")
            return NextResponse.json({
                success: true,
                message: "Generation was successfully stopped.",
            })
        } catch (fetchError) {
            console.error("Backend server connection error:", fetchError)

            // Even if backend server doesn't respond, respond with success to client
            return NextResponse.json({
                success: true,
                warning: "Backend server connection failed, stopped locally only",
                message: "Failed to connect to backend server, but UI will update to stopped state.",
            })
        }
    } catch (error) {
        console.error("API processing error:", error)
        return NextResponse.json(
            {
                success: true, // Always respond with success to client
                error: `Error processing request: ${error instanceof Error ? error.message : "Unknown error"}`,
                message: "Error occurred while processing stop request. UI will update to stopped state.",
            },
            { status: 200 }, // Respond with 200 to client
        )
    }
}
