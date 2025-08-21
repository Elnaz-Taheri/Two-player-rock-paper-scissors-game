import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Landmark indices for finger tips
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        handData = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                handData.append(lmList)
        return handData

    def fingersUp(self, lmList):
        fingers = []
        if not lmList: 
            return []

        # Thumb (simple logic assuming right hand, might need refinement)
        if lmList[self.tipIds[0]][1] > lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if lmList[self.tipIds[id]][2] < lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def get_gesture(fingers):
    if fingers == [0,0,0,0,0]:
        return "Rock"
    elif fingers == [1,1,1,1,1]:
        return "Paper"
    elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
        return "Scissors"
    else:
        return "Unknown"

def decide_winner(move1, move2):
    if move1 == move2:
        return "Draw"
    elif (move1 == "Rock" and move2 == "Scissors") or \
         (move1 == "Paper" and move2 == "Rock") or \
         (move1 == "Scissors" and move2 == "Paper"):
        return "Player 1 Wins"
    elif move1 == "Unknown" or move2 == "Unknown":
        return "Waiting for valid moves..."
    else:
        return "Player 2 Wins"

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=2)

    move_player1 = "Waiting..."
    move_player2 = "Waiting..."
    result = ""

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        hands = detector.findPosition(img)
        height, width, _ = img.shape

        # Reset moves every frame
        move_player1 = "Waiting..."
        move_player2 = "Waiting..."

        for hand_lmList in hands:
            fingers = detector.fingersUp(hand_lmList)
            move = get_gesture(fingers)

            # Determine if hand is left or right based on x coordinate of wrist (landmark 0)
            wrist_x = hand_lmList[0][1]

            if wrist_x < width // 2:
                move_player1 = move
                cv2.putText(img, f"P1: {move_player1}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                move_player2 = move
                cv2.putText(img, f"P2: {move_player2}", (width - 250, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Decide winner only if both moves are valid and known
        if move_player1 != "Waiting..." and move_player2 != "Waiting...":
            result = decide_winner(move_player1, move_player2)
        else:
            result = "Waiting for both players..."
        cv2.line(img,(width//2,0),(width//2,height),(0,0,255))
        cv2.putText(img, f"Result: {result}", (width // 4, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Rock Paper Scissors - Two Players", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
