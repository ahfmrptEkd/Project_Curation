import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_exceptions():
    """Test the custom exceptions."""
    print("🧪 예외 클래스 테스트")
    print("-" * 30)

    try:
        from src.components.utils.exceptions import APIError
        from src.components.utils.exceptions import ColdStartError
        from src.components.utils.exceptions import DataError
        from src.components.utils.exceptions import NewUserError

        # Test exception creation and handling
        try:
            raise APIError("테스트 API 에러", "API001")
        except APIError as e:
            print(f"✅ APIError 처리: {e}")

        try:
            raise DataError("테스트 데이터 에러", "DATA001")
        except DataError as e:
            print(f"✅ DataError 처리: {e}")

        try:
            raise NewUserError("신규 사용자")
        except (
            ColdStartError
        ) as e:  # NewUserError는 ColdStartError의 하위 클래스
            print(f"✅ ColdStartError 계층 처리: {e}")

        return True

    except Exception as e:
        print(f"❌ 예외 테스트 실패: {e}")
        return False


def test_logging():
    """Test the simple logging system."""
    print("\n📝 로깅 시스템 테스트")
    print("-" * 30)

    try:
        from src.components.utils.exceptions import APIError
        from src.components.utils.logger import log_error
        from src.components.utils.logger import log_info
        from src.components.utils.logger import setup_logger

        # Setup logger
        logger = setup_logger("test_logger")

        # Test basic logging
        log_info(logger, "테스트 정보 메시지", "TEST")

        # Test error logging
        try:
            raise APIError("테스트용 API 에러")
        except Exception as e:
            log_error(logger, e, "ERROR_TEST")

        print("✅ 로깅 시스템 정상 작동")
        return True

    except Exception as e:
        print(f"❌ 로깅 테스트 실패: {e}")
        return False


def test_config():
    """Test the simple configuration system."""
    print("\n⚙️ 설정 시스템 테스트")
    print("-" * 30)

    try:
        from src.components.config.config import get_app_config
        from src.components.config.config import get_logging_config
        from src.components.config.config import load_config

        # Test config loading
        config = load_config()
        print(f"✅ 설정 로드 성공: {list(config.keys())}")

        # Test specific config getters
        app_config = get_app_config()
        print(f"✅ 앱 설정: {app_config.get('name')}")

        logging_config = get_logging_config()
        print(f"✅ 로깅 설정: Level {logging_config.get('level')}")

        return True

    except Exception as e:
        print(f"❌ 설정 테스트 실패: {e}")
        return False


def test_integrated_usage():
    """Test integrated usage of the error handling system."""
    print("\n🔧 통합 사용 테스트")
    print("-" * 30)

    try:
        from src.components.config.config import get_logging_config
        from src.components.utils.exceptions import APIError
        from src.components.utils.exceptions import DataError
        from src.components.utils.logger import log_error
        from src.components.utils.logger import setup_logger

        # Setup logger with config
        log_config = get_logging_config()
        logger = setup_logger(
            "integrated_test",
            log_level=log_config.get("level", "INFO"),
            log_to_file=log_config.get("to_file", True),
            log_dir=log_config.get("log_dir", "logs"),
        )

        # Simulate some application errors
        def risky_api_call():
            raise APIError("외부 API 연결 실패")

        def risky_data_operation():
            raise DataError("데이터 검증 실패")

        # Handle errors with logging
        try:
            risky_api_call()
        except APIError as e:
            log_error(logger, e, "API_CALL")
            print("✅ API 에러 처리 및 로깅 완료")

        try:
            risky_data_operation()
        except DataError as e:
            log_error(logger, e, "DATA_OP")
            print("✅ 데이터 에러 처리 및 로깅 완료")

        print("✅ 통합 사용 테스트 성공")
        return True

    except Exception as e:
        print(f"❌ 통합 테스트 실패: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 단순화된 에러 핸들링 시스템 테스트")
    print("=" * 50)

    tests = [
        ("예외 클래스", test_exceptions),
        ("로깅 시스템", test_logging),
        ("설정 시스템", test_config),
        ("통합 사용", test_integrated_usage),
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
    print("=" * 50)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"{test_name}: {status}")

    print(f"\n총 {total}개 테스트 중 {passed}개 성공")

    if passed == total:
        print("🎉 모든 테스트 성공!")
        print("📝 로그 파일은 logs/ 폴더에서 확인할 수 있습니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
