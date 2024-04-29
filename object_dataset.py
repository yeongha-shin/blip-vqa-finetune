import cv2
import os
import json
import argparse
import shutil
import scipy.io as sio
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Save video frames and JSON data")
    parser.add_argument("--video_path", required=True, help="Input video path")
    parser.add_argument("--object_path", required=True, help="video object path")
    parser.add_argument("--output_path", required=True, help="Output dataset path")
    parser.add_argument("--total_data_path", required=True, help="Total JSON Lines path")
    args = parser.parse_args()
    return args

def clear_folder(folder_path):
    """Remove all files in the specified folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def count_object_types(object_data):
    # 모든 요소를 추출하고 플래트(flatten) 처리
    flattened_data = np.concatenate(object_data.flatten())

    # 각 요소의 값 추출
    # type_list = [item[0] for item in flattened_data]
    type_list = flattened_data

    # 유니크한 요소와 그 개수 계산
    unique_types, counts = np.unique(type_list, return_counts=True)

    # 결과 출력
    result = dict(zip(unique_types, counts))
    return result

def save_frames_and_json(video_path, output_path, total_data_path, object_path):
    # 비디오를 읽어오기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    frame_index = 0

    arr = sio.loadmat(object_path)
    object_data = arr['structXML']

    # object_data[0][frame_index][3] : Moving Stationary
    # object_data[0][frame_index][4] : Vessel/ship, Speed boat, buoy
    # object_data[0][frame_index][5] : Far, Near,

    # 파일을 추가 모드로 열기 (기존 내용을 유지하면서 새 데이터 추가)
    with open(total_data_path, 'a') as total_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임을 저장할 폴더 생성
            folder_name = os.path.join(output_path, str(frame_index))
            if os.path.exists(folder_name):
                clear_folder(folder_name)  # 폴더가 이미 존재하면 내용 삭제
            else:
                os.makedirs(folder_name, exist_ok=True)

            # 이미지 파일 저장
            image_path = os.path.join(folder_name, "image.png")
            cv2.imwrite(image_path, frame)

            # Object 정보 불러오기

            detections = object_data[0][frame_index][6]

            for det in detections:
                # Append the converted detection to dets list
                print("box = ", det[0], det[1])

            # count object
            type_result = count_object_types(object_data[0][frame_index][4])
            # print(type_result['Buoy'])

            # 사전의 키 리스트 생성
            keys_list = list(type_result.keys())

            # 모든 키-값 쌍을 문장으로 변환
            sentences = []
            for key in keys_list:
                count = type_result[key]
                if count == 1:
                    sentences.append(f"{count} {key}")
                else:
                    sentences.append(f"{count} {key}")

            # 문장을 연결하여 하나의 줄글로 생성
            full_sentence = "There are " + ", ".join(sentences[:-1]) + ", and " + sentences[-1] + "."
            print(full_sentence)

            # JSON 파일 생성 및 저장
            data = {
                "question": "What kinds of objects are there?",
                "answer": full_sentence,
                "ques_type": "explain",
                "grade": "0",
                "label": "0"
            }

            json_path = os.path.join(folder_name, "data.json")
            with open(json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            # 1. Object Existance
            data = {
                "question": "What kinds of objects are there?",
                "answer": full_sentence,
                "ques_type": "explain",
                "grade": "0",
                "label": "0",
                "pid": str(frame_index)  # 프레임 인덱스를 pid로 추가
            }

            # total_train.jsonl에 데이터 추가 (각 줄에 JSON 객체)
            json_line = json.dumps(data) + '\n'
            total_file.write(json_line)

            # 2. Object Near / Far
            close_result = count_object_types(object_data[0][frame_index][5])

            # "Near" 객체의 개수를 확인
            near_count = close_result.get('Near', 0)

            # "Near" 객체의 개수에 따라 메시지를 조건부로 출력
            if near_count > 0:
                full_sentence = f"No, there are {near_count} near objects, which makes it dangerous."
            else:
                full_sentence = "Yes, there are no near objects, which makes it safe."

            data = {
                "question": "Is it safe now?",
                "answer": full_sentence,
                "ques_type": "explain",
                "grade": "0",
                "label": "0",
                "pid": str(frame_index)  # 프레임 인덱스를 pid로 추가
            }

            # total_train.jsonl에 데이터 추가 (각 줄에 JSON 객체)
            json_line = json.dumps(data) + '\n'
            total_file.write(json_line)

            frame_index += 1

    cap.release()
    print("Finished processing all frames")

# 비디오 파일 경로 입력받기
args = parse_args()
save_frames_and_json(args.video_path, args.output_path, args.total_data_path, args.object_path)
