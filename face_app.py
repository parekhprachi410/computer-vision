import cv2
import tkinter as tk
from PIL import Image, ImageTk
from datetime import datetime

class FaceApp:

    def __init__(self, root):

        self.root = root
        self.root.title("AI Face Detection App")
        self.root.geometry("1000x750")
        self.root.configure(bg="#222222")

        self.cap = None
        self.running = False
        self.current_frame = None

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        title = tk.Label(
            root,
            text="REAL-TIME FACE DETECTION",
            font=("Arial",22,"bold"),
            fg="white",
            bg="#222222"
        )
        title.pack(pady=10)

        self.video_label = tk.Label(root,bg="black")
        self.video_label.pack(pady=20)

        button_frame = tk.Frame(root,bg="#222222")
        button_frame.pack()

        tk.Button(
            button_frame,
            text="Start Camera",
            command=self.start,
            bg="green",
            fg="white",
            font=("Arial",14)
        ).grid(row=0,column=0,padx=10)

        tk.Button(
            button_frame,
            text="Capture Photo",
            command=self.capture_photo,
            bg="blue",
            fg="white",
            font=("Arial",14)
        ).grid(row=0,column=1,padx=10)

        tk.Button(
            button_frame,
            text="Stop Camera",
            command=self.stop,
            bg="red",
            fg="white",
            font=("Arial",14)
        ).grid(row=0,column=2,padx=10)

    def start(self):

        if not self.running:

            self.cap = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            self.running = True
            self.update_frame()

    def update_frame(self):

        if self.running:

            ret, frame = self.cap.read()

            if ret:

                self.current_frame = frame.copy()

                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY
                )

                faces = self.face_detector.detectMultiScale(
                    gray,
                    1.1,
                    5
                )

                for (x,y,w,h) in faces:

                    cv2.rectangle(
                        frame,
                        (x,y),
                        (x+w,y+h),
                        (0,255,0),
                        3
                    )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                img = Image.fromarray(rgb)
                img = img.resize((850,550))

                photo = ImageTk.PhotoImage(img)

                self.video_label.configure(image=photo)
                self.video_label.image = photo

            self.root.after(15,self.update_frame)

    def capture_photo(self):

        if self.current_frame is not None:

            filename = datetime.now().strftime(
                "capture_%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(
                filename,
                self.current_frame
            )

            print(
                f"Saved: {filename}"
            )

    def stop(self):

        self.running = False

        if self.cap:
            self.cap.release()

root = tk.Tk()
app = FaceApp(root)
root.mainloop()