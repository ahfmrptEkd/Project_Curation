import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_exceptions():
    """Test the custom exceptions with standardized error codes."""
    print("🧪 예외 클래스 테스트 (표준화된 에러 코드)")
    print("-" * 50)

    try:
        from src.components.utils.exceptions import APIError
        from src.components.utils.exceptions import ColdStartError
        from src.components.utils.exceptions import DataError
        from src.components.utils.exceptions import ErrorCodes
        from src.components.utils.exceptions import NewUserError
        from src.components.utils.exceptions import OpenAIAPIError

        # Test exception creation and handling with error codes
        try:
            raise APIError("테스트 API 에러")
        except APIError as e:
            print(f"✅ APIError 처리: {e}")
            print(f"   에러 코드: {e.error_code}")
            print(f"   에러 딕셔너리: {e.to_dict()}")

        try:
            raise DataError("테스트 데이터 에러")
        except DataError as e:
            print(f"✅ DataError 처리: {e}")
            print(f"   에러 코드: {e.error_code}")

        try:
            raise OpenAIAPIError(
                "OpenAI API 연결 실패", details={"status_code": 429}
            )
        except OpenAIAPIError as e:
            print(f"✅ OpenAIAPIError 처리: {e}")
            print(f"   에러 코드: {e.error_code}")
            print(f"   세부 정보: {e.details}")

        try:
            raise NewUserError("신규 사용자")
        except ColdStartError as e:
            print(f"✅ ColdStartError 계층 처리: {e}")
            print(f"   에러 코드: {e.error_code}")

        # Test error codes
        print("\n📋 에러 코드 상수 테스트:")
        print(f"   DATA_001: {ErrorCodes.DATA_001}")
        print(f"   API_001: {ErrorCodes.API_001}")
        print(f"   CONFIG_001: {ErrorCodes.CONFIG_001}")

        return True

    except Exception as e:
        print(f"❌ 예외 테스트 실패: {e}")
        return False


def test_logging_security():
    """Test the enhanced logging system with security features."""
    print("\n🔒 보안 로깅 시스템 테스트")
    print("-" * 50)

    try:
        from src.components.utils.logger import log_info
        from src.components.utils.logger import sanitize_log_message
        from src.components.utils.logger import setup_logger

        # Test sensitive data masking
        sensitive_messages = [
            "API 키: api_key=sk-1234567890abcdef",
            "사용자 이메일: user@example.com",
            "비밀번호: password=secret123",
            "토큰: access_token=abcdef123456789",
            "설정: {openai_key: sk-proj-1234567890abcdef, user_id: user123456789012345}",
        ]

        print("🔍 민감한 정보 마스킹 테스트:")
        for msg in sensitive_messages:
            masked = sanitize_log_message(msg)
            print(f"   원본: {msg}")
            print(f"   마스킹: {masked}")
            print()

        # Test logger with security enabled
        logger = setup_logger("security_test", enable_security=True)
        log_info(
            logger,
            "API 키 테스트: api_key=sk-1234567890abcdef",
            "SECURITY_TEST",
        )

        # Test logger with security disabled
        logger_no_security = setup_logger(
            "no_security_test", enable_security=False
        )
        log_info(logger_no_security, "보안 비활성화 테스트", "NO_SECURITY")

        print("✅ 보안 로깅 시스템 정상 작동")
        return True

    except Exception as e:
        print(f"❌ 보안 로깅 테스트 실패: {e}")
        return False


def test_logging_rotation():
    """Test log file rotation feature."""
    print("\n🔄 로그 파일 순환 테스트")
    print("-" * 50)

    try:
        from src.components.utils.logger import log_info
        from src.components.utils.logger import setup_logger

        # Test logger with rotation enabled
        logger = setup_logger(
            "rotation_test",
            enable_rotation=True,
            max_file_size=1024,  # 1KB for testing
            backup_count=3,
        )

        # Generate some log messages
        for i in range(20):
            log_info(
                logger,
                f"로그 순환 테스트 메시지 {i+1} - 이것은 로그 파일 순환 기능을 테스트하기 위한 긴 메시지입니다.",
                "ROTATION_TEST",
            )

        print("✅ 로그 파일 순환 테스트 완료")
        print("   logs/rotation_test.log 및 백업 파일들을 확인하세요.")
        return True

    except Exception as e:
        print(f"❌ 로그 파일 순환 테스트 실패: {e}")
        return False


def test_structured_logging():
    """Test structured logging feature."""
    print("\n📊 구조화된 로깅 테스트")
    print("-" * 50)

    try:
        from src.components.utils.logger import log_structured
        from src.components.utils.logger import setup_logger

        logger = setup_logger("structured_test")

        # Test structured logging
        log_structured(
            logger,
            "api_call",
            {
                "endpoint": "/recommendations",
                "method": "POST",
                "status_code": 200,
                "duration_ms": 150,
                "user_id": "user123",
            },
            "API_CALL",
        )

        log_structured(
            logger,
            "error_event",
            {
                "error_type": "APIError",
                "error_code": "API_001",
                "message": "OpenAI API 연결 실패",
                "retry_count": 3,
            },
            "ERROR_EVENT",
        )

        print("✅ 구조화된 로깅 테스트 완료")
        return True

    except Exception as e:
        print(f"❌ 구조화된 로깅 테스트 실패: {e}")
        return False


def test_config_enhancements():
    """Test enhanced configuration system."""
    print("\n⚙️ 향상된 설정 시스템 테스트")
    print("-" * 50)

    try:
        from src.components.config.config import get_logging_config
        from src.components.config.config import get_performance_config
        from src.components.config.config import get_security_config

        # Test enhanced logging config
        logging_config = get_logging_config()
        print("✅ 로깅 설정 로드 성공:")
        print(f"   레벨: {logging_config.get('level')}")
        print(f"   파일 순환: {logging_config.get('enable_rotation')}")
        print(f"   보안 활성화: {logging_config.get('enable_security')}")
        print(f"   최대 파일 크기: {logging_config.get('max_file_size')} bytes")

        # Test security config
        security_config = get_security_config()
        print("\n✅ 보안 설정 로드 성공:")
        print(f"   보안 활성화: {security_config.get('enable_security')}")
        print(
            f"   민감한 데이터 마스킹: {security_config.get('sensitive_data_masking')}"
        )

        # Test performance config
        performance_config = get_performance_config()
        print("\n✅ 성능 설정 로드 성공:")
        print(f"   성능 로깅: {performance_config.get('performance_logging')}")
        print(
            f"   느린 쿼리 임계값: {performance_config.get('slow_query_threshold')}ms"
        )

        return True

    except Exception as e:
        print(f"❌ 향상된 설정 테스트 실패: {e}")
        return False


def test_integrated_usage():
    """Test integrated usage of the enhanced error handling system."""
    print("\n🔧 통합 사용 테스트 (향상된 버전)")
    print("-" * 50)

    try:
        from src.components.config.config import get_logging_config
        from src.components.utils.exceptions import DataError
        from src.components.utils.exceptions import OpenAIAPIError
        from src.components.utils.logger import log_error
        from src.components.utils.logger import log_structured
        from src.components.utils.logger import setup_logger

        # Setup logger with enhanced config
        log_config = get_logging_config()
        logger = setup_logger(
            "integrated_test_enhanced",
            log_level=log_config.get("level", "INFO"),
            log_to_file=log_config.get("to_file", True),
            log_dir=log_config.get("log_dir", "logs"),
            enable_rotation=log_config.get("enable_rotation", True),
            max_file_size=log_config.get("max_file_size", 10 * 1024 * 1024),
            backup_count=log_config.get("backup_count", 5),
            enable_security=log_config.get("enable_security", True),
        )

        # Simulate enhanced application errors
        def risky_api_call():
            raise OpenAIAPIError(
                "OpenAI API 연결 실패 - API 키 문제",
                details={
                    "api_key": "sk-proj-1234567890abcdef",
                    "status_code": 401,
                },
            )

        def risky_data_operation():
            raise DataError(
                "데이터 검증 실패 - 사용자 정보 오류",
                details={
                    "user_email": "user@example.com",
                    "validation_errors": ["이메일 형식", "필수 필드 누락"],
                },
            )

        # Handle errors with enhanced logging
        try:
            risky_api_call()
        except OpenAIAPIError as e:
            log_error(logger, e, "API_CALL")
            # Structured logging for error analysis
            log_structured(
                logger,
                "api_error",
                {
                    "error_code": e.error_code,
                    "error_type": type(e).__name__,
                    "details": e.details,
                    "context": "API_CALL",
                },
            )
            print("✅ 향상된 API 에러 처리 및 로깅 완료")

        try:
            risky_data_operation()
        except DataError as e:
            log_error(logger, e, "DATA_OP")
            # Structured logging for error analysis
            log_structured(
                logger,
                "data_error",
                {
                    "error_code": e.error_code,
                    "error_type": type(e).__name__,
                    "details": e.details,
                    "context": "DATA_OP",
                },
            )
            print("✅ 향상된 데이터 에러 처리 및 로깅 완료")

        print("✅ 통합 사용 테스트 성공 (향상된 버전)")
        return True

    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False


def main():
    """Run all tests including new enhanced features."""
    print("🧪 향상된 에러 핸들링 시스템 테스트")
    print("=" * 60)

    tests = [
        ("예외 클래스 (표준화된 에러 코드)", test_exceptions),
        ("보안 로깅 시스템", test_logging_security),
        ("로그 파일 순환", test_logging_rotation),
        ("구조화된 로깅", test_structured_logging),
        ("향상된 설정 시스템", test_config_enhancements),
        ("통합 사용 (향상된 버전)", test_integrated_usage),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 실행 중 오류: {e}")
            results.append((test_name, False))

    # Summary
    print("\n📊 테스트 결과")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")

    print(f"\n총 {total}개 테스트 중 {passed}개 성공")

    if passed == total:
        print("🎉 모든 테스트 성공!")
        print("📝 향상된 기능들:")
        print("   - 민감한 정보 마스킹")
        print("   - 로그 파일 순환")
        print("   - 표준화된 에러 코드")
        print("   - 구조화된 로깅")
        print("   - 향상된 설정 시스템")
        print("📂 로그 파일은 logs/ 폴더에서 확인할 수 있습니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
