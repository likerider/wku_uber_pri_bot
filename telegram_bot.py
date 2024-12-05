from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ConversationHandler,
    ContextTypes
)
import torch
from model import Model
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any
import httpx
import certifi
import ssl
import urllib3
import asyncio
from asyncio import Event
import tracemalloc
import traceback
import platform
import gc
import warnings
import torch.nn as nn
import telegram

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 환경변수 로드
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

# 상태 정의
DISTANCE, CAB_TYPE, WEATHER, SURGE = range(4)

# cleanup 함수 정의
def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class TaxiPriceBot:
    def __init__(self):
        # 로깅 설정
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO,
            filename='bot.log'
        )
        self.logger = logging.getLogger(__name__)
        
        # 현재 파일의 디렉토리 경로 가져오기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'model', 'model.pth')
        
        # 모델 로드 최적화
        self.model = Model(
            in_features=14,
            hidden_features=[512, 512, 256, 256],
            out_features=1,
            batch_norm=nn.BatchNorm1d,
            dropout=0.1,
            init_weights=False
        )
        
        # 모델 파일 존재 확인
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.logger.info(f"모델을 성공적으로 로드했습니다: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise
        
        # 메모리 정리
        cleanup()
        
        # 사용자 데이터 저장
        self.user_data: Dict[int, Dict[str, Any]] = {}
        
        # 유효한 입력값 정의
        self.valid_cabs = {
            'Black': '블랙',
            'Black SUV': '블랙 SUV', 
            'Lux': '럭스',
            'Lux Black': '럭스 블랙',
            'Lux Black XL': '럭스 블랙 XL',
            'Lyft': '리프트',
            'Lyft XL': '리프트 XL',
            'Shared': '쉐어드',
            'UberPool': '우버풀',
            'UberX': '우버X',
            'UberXL': '우버XL',
            'WAV': 'WAV'
        }
        
        self.valid_weather = {
            '맑음 (낮)': 'clear-day',
            '맑음 (밤)': 'clear-night',
            '흐림': 'cloudy',
            '안개': 'fog',
            '구름 조금 (낮)': 'partly-cloudy-day',
            '구름 조금 (밤)': 'partly-cloudy-night',
            '비': 'rain'
        }
        
        # 텔레그램 봇 설정
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise ValueError("Telegram Bot Token이 설정되지 않았습니다.")
            
        # 비동기 처리 최적화를 위한 설정
        self.application = (
            ApplicationBuilder()
            .token(token)
            .connection_pool_size(8)  # 연결 풀 크기 증가
            .connect_timeout(30.0)    # 연결 타임아웃 감소
            .read_timeout(30.0)       # 읽기 타임아웃 감소
            .write_timeout(30.0)      # 쓰기 타임아웃 감소
            .pool_timeout(30.0)       # 풀 타임아웃 감소
            .build()
        )
        
        # 대화 핸들러 설정
        conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler('start', self.start),
                CommandHandler('help', self.help)
            ],
            states={
                DISTANCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.distance)],
                CAB_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.cab_type)],
                WEATHER: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.weather)],
                SURGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.surge)]
            },
            fallbacks=[
                CommandHandler('cancel', self.cancel),
                CommandHandler('help', self.help)
            ],
            per_chat=True,  # 채팅별로 독립적인 상태 유지
            per_user=True   # 사용자별로 독립적인 상태 유지
        )
        
        self.application.add_handler(conv_handler)
        self.application.add_error_handler(self.error)
    
    async def help(self, update: Update, context: ContextTypes) -> None:
        """도움말 표시"""
        help_text = (
            "🚕 택시 요금 예측 봇 사용법 🚕\n\n"
            "1. /start 명령어로 예측 시작\n"
            "2. 이동 거리를 킬로미터(km) 단위로 입력\n"
            "3. 량 종류 선택\n"
            "4. 날씨 선택\n"
            "5. 수요배수 입력 (1.0~3.0)\n\n"
            "📱 사용 가능한 명령어:\n"
            "/start - 요금 예측 시작\n"
            "/cancel - 현재 진행 중인 예측 취소\n"
            "/help - 도움말 표시\n\n"
            "🚗 지원되는 차량 종류:\n"
        )
        
        for cab, desc in self.valid_cabs.items():
            help_text += f"• {cab}: {desc}\n"
            
        await update.message.reply_text(help_text)
        return ConversationHandler.END if context.user_data else None
    
    async def start(self, update: Update, context: ContextTypes) -> int:
        """대화 시작"""
        user = update.message.from_user
        self.logger.info(f"사용자 {user.first_name} ({user.id})가 예측을 시작했습니다.")
        
        await update.message.reply_text(
            f'안녕하세요 {user.first_name}님! 택시 요금 예측 봇입니다. 🚕\n'
            '요금을 예측하기 위해 몇 가지 정보가 필요합니다.\n\n'
            '먼저, 이동하실 거리(km)를 입력해주세요.\n'
            '(예: 5.2)\n\n'
            '도움말을 보시려면 /help를 입력해주세요.\n'
            '취소시려면 /cancel을 입력주세요.'
        )
        return DISTANCE
    
    async def distance(self, update: Update, context: ContextTypes) -> int:
        """거리 입력 처리"""
        try:
            distance = float(update.message.text)
            if distance <= 0:
                await update.message.reply_text('❌ 거리는 0보다 커야 합니다. 다시 입력해주세요.')
                return DISTANCE
            elif distance > 100:
                await update.message.reply_text('❌ 100km 이하의 거리를 입력해주세요.')
                return DISTANCE
                
            self.user_data[update.message.chat_id] = {'distance': distance}
            
            # 차량 종류 선택 키보드 생성
            reply_keyboard = [
                list(self.valid_cabs.keys())[i:i+2] 
                for i in range(0, len(self.valid_cabs), 2)
            ]
            
            await update.message.reply_text(
                '🚗 차량 종류를 선택해주세요:',
                reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
            )
            return CAB_TYPE
            
        except ValueError:
            await update.message.reply_text('❌ 올바른 숫자를 입력해주세요.')
            return DISTANCE
    
    async def cab_type(self, update: Update, context: ContextTypes) -> int:
        """차량 종류 선택 처리"""
        cab = update.message.text
        
        if cab not in self.valid_cabs:
            await update.message.reply_text('❌ 올바른 차량 종류를 선택해주세요.')
            return CAB_TYPE
            
        self.user_data[update.message.chat_id]['cab_type'] = cab
        
        # 날씨 선택 키보드
        reply_keyboard = [
            list(self.valid_weather.keys())[i:i+2] 
            for i in range(0, len(self.valid_weather), 2)
        ]
        
        await update.message.reply_text(
            '🌤 날씨를 선택해주세요:',
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return WEATHER
    
    async def weather(self, update: Update, context: ContextTypes) -> int:
        """날씨 선택 처리"""
        weather = update.message.text
        
        if weather not in self.valid_weather:
            await update.message.reply_text('❌ 올바른 날씨를 선택해주세요.')
            return WEATHER
            
        self.user_data[update.message.chat_id]['weather'] = weather
        
        await update.message.reply_text(
            '📈 수요배수를 입력해주세요 (1.0~3.0):\n'
            '1.0: 보통\n'
            '2.0: 혼잡\n'
            '3.0: 매우 혼잡',
            reply_markup=ReplyKeyboardRemove()
        )
        return SURGE

    async def surge(self, update: Update, context: ContextTypes) -> int:
        """수요배수 입력 및 최종 예측"""
        try:
            surge = float(update.message.text)
            if surge < 1.0 or surge > 3.0:
                await update.message.reply_text('❌ 1.0에서 3.0 사이의 값을 입력해주세요.')
                return SURGE

            chat_id = update.message.chat_id
            user_input = self.user_data[chat_id]
            user_input['surge'] = surge

            # 입력 데이터 처리 및 예측
            with torch.no_grad():
                input_tensor = self.prepare_input(
                    user_input['distance'],
                    user_input['cab_type'],
                    self.valid_weather[user_input['weather']],
                    user_input['surge']
                )
                prediction = self.model(input_tensor)
            
            # 로그 기록
            self.logger.info(
                f"예측 완료 - 거리: {user_input['distance']}, "
                f"차량: {user_input['cab_type']}, "
                f"날씨: {user_input['weather']}, "
                f"수요배수: {user_input['surge']}, "
                f"예상 요금: ${float(prediction):.2f}"
            )
            
            await update.message.reply_text(
                f'💰 예상 요금: ${float(prediction):.2f}\n\n'
                f'📋 입력하신 정보:\n'
                f'• 거리: {user_input["distance"]}km\n'
                f'• 차량: {user_input["cab_type"]}\n'
                f'• 날씨: {user_input["weather"]}\n'
                f'• 수요배수: {user_input["surge"]}\n\n'
                f'새로운 예측을 하시려면 /start를 입력해주세요.',
                reply_markup=ReplyKeyboardRemove()
            )
            
            # 사용자 데이터 삭제
            del self.user_data[chat_id]
            return ConversationHandler.END
            
        except ValueError:
            await update.message.reply_text('❌ 올바른 숫자를 입력해주세요.')
            return SURGE
        except Exception as e:
            self.logger.error(f"예측 중 오류 발생: {str(e)}")
            await update.message.reply_text(
                '❌ 죄송합니다. 예측 중 오류가 발생했습니다.\n'
                '다시 시도하려면 /start를 입력해주세요.',
                reply_markup=ReplyKeyboardRemove()
            )
            return ConversationHandler.END
    
    def prepare_input(self, distance: float, cab_type: str, weather: str, surge: float) -> torch.Tensor:
        """모델 입력 데이터 준비"""
        try:
            # 초기 벡터 생성 (모두 0으로 초기화)
            input_vector = [0] * 14  # in_features=14
            
            # 1. distance 설정
            input_vector[12] = distance  # distance는 13번째 위치
            
            # 2. surge_multiplier 설정
            input_vector[13] = surge  # surge_multiplier는 14번째 위치
            
            # 3. 차량 종류 원-핫 인코딩 (12개 종류)
            cab_type_map = {
                'Black': 0,
                'Black SUV': 1,
                'Lux': 2,
                'Lux Black': 3,
                'Lux Black XL': 4,
                'Lyft': 5,
                'Lyft XL': 6,
                'Shared': 7,
                'UberPool': 8,
                'UberX': 9,
                'UberXL': 10,
                'WAV': 11
            }
            
            if cab_type not in cab_type_map:
                raise ValueError(f"지원하지 않는 차량 종류: {cab_type}")
                
            input_vector[cab_type_map[cab_type]] = 1
            
            # 텐서로 변환
            input_tensor = torch.tensor([input_vector], dtype=torch.float32)
            
            return input_tensor
            
        except Exception as e:
            self.logger.error(f"입력 데이터 준비 중 오류 발생: {str(e)}")
            raise
    
    async def cancel(self, update: Update, context: ContextTypes) -> int:
        """대화 취소"""
        user = update.message.from_user
        chat_id = update.message.chat_id
        
        self.logger.info(f"사용자 {user.first_name} ({user.id})가 예측을 취소했습니다.")
        
        if chat_id in self.user_data:
            del self.user_data[chat_id]
            
        await update.message.reply_text(
            '❌ 요금 예측이 취소되었습니다.\n'
            '다시 시작하려면 /start를 입력해주세요.',
            reply_markup=ReplyKeyboardRemove()
        )
        return ConversationHandler.END
    
    async def error(self, update: Update, context: ContextTypes) -> None:
        """에러 처리"""
        self.logger.error(f'Update {update} caused error {context.error}')
        
        if update and hasattr(update, 'message'):
            await update.message.reply_text(
                '❌ 죄송합니다. 오류가 발생했습니다.\n'
                '다시 시도하려면 /start를 입력해주세요.'
            )
    
    def run(self):
        """봇 실행"""
        try:
            self.logger.info("🚀 봇이 시작되었습니다.")
            self.application.run_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True  # 오래된 업데이트 무시
            )
        except Exception as e:
            self.logger.error(f"봇 실행 중 오류 발생: {str(e)}")
            raise

if __name__ == '__main__':
    # 메모리 트레이싱 시작
    tracemalloc.start()
    
    try:
        # 봇 인스턴스 생성 및 실행
        bot = TaxiPriceBot()
        bot.run()
        
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 종료되었습니다.")
    except Exception as e:
        logging.error(f"프로그램 실행 중 오류 발생: {str(e)}")
        print(f"오류가 발생했습니다: {str(e)}")
    finally:
        # 정리 작업
        cleanup()
        tracemalloc.stop()

# 메모리 최적화를 위한 설정
gc.enable()
